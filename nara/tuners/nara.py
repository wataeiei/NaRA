# coding=utf-8
# License: Apache 2.0
from __future__ import annotations

import math
import re
import os
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union, Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import weakref
from transformers.pytorch_utils import Conv1D

# ----- PEFT/Utils hooks -----
from ..import_utils import is_bnb_available
from ..utils import (
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
    PeftConfig,
    PeftType,
    _freeze_adapter,
    _get_submodules,
    transpose,
)

if is_bnb_available():
    import bitsandbytes as bnb


# ---------------------------------------------------------------------
# Helpers & Embedding Classes
# ---------------------------------------------------------------------
def _as_tensor_like(x, ref: torch.Tensor, *, dtype=None):
    """Cast x to a tensor on ref's device/dtype (unless dtype is provided)."""
    if torch.is_tensor(x):
        # We need to ensure the tensor has a batch dimension (at least 1)
        x = x.view(-1) if x.dim() == 0 else x # Ensure at least 1D
        return x.to(device=ref.device, dtype=dtype or ref.dtype)
    
    # x is float/int, convert to tensor with batch size 1
    return torch.tensor([x], device=ref.device, dtype=dtype or ref.dtype)


class GaussianFourierProjection(nn.Module):
    """
    Gaussian Fourier embeddings for continuous inputs (Fixed, Non-Learnable).
    """
    def __init__(self, embed_dim: int, scale: float = 16.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Random weight matrix B, fixed (not learnable)
        # Input dim is 1, output features is embed_dim // 2 (for sin/cos pair)
        if embed_dim % 2 != 0:
            raise ValueError(f"GaussianFourierProjection embed_dim must be even, got {embed_dim}")
            
        # Using scale=16.0 by default provides good coverage for [0,1] inputs
        self.register_buffer("W", torch.randn(1, embed_dim // 2) * scale)

    def forward(self, x: torch.Tensor):
        # x shape: [Batch, 1] or [Batch]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # x_proj = x @ W (Batch, dim/2)
        # Compute in float32 for numerical stability (sin/cos) and to match x.float()
        x_proj = x.float() @ self.W.to(dtype=torch.float32)
        
        # [sin(2pi W x), cos(2pi W x)]
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        
        # Cast back to the buffer's dtype (e.g., float16) to match the rest of the model
        return out.to(dtype=self.W.dtype)


class MLPEmbedding(nn.Module):
    """
    Simple MLP embedding for continuous inputs (Learnable).
    Linear(1 -> dim) -> SiLU
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(1, embed_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor):
        # x shape: [Batch, 1] or [Batch]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        
        # Cast to same dtype as weights for calculation
        x = x.to(dtype=self.proj.weight.dtype)
        return self.act(self.proj(x))


class RawEmbedding(nn.Module):
    """
    Pass-through for raw scalar inputs.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x: torch.Tensor):
        # x shape: [Batch, 1] or [Batch]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return x


class NARAMapper(nn.Module):
    """
    The Single Global FNN that computes C for NARA.
    Structure: Linear -> SiLU -> Linear -> SiLU -> Linear
    (Removed LayerNorm and Dropout as requested).
    """
    def __init__(self, 
                 r_ab: int, 
                 input_dim: int, 
                 fnn_hidden_size_1: int, 
                 fnn_hidden_size_2: int,
                 init_c: str = "zero_last"):
        super().__init__()
        
        output_dim = r_ab * r_ab
        
        if input_dim <= 0:
            raise ValueError("NARAMapper cannot be initialized with zero input dimensions.")

        self.init_c = init_c
        
        modules = []
        
        # Hidden Layer 1: Linear -> Activation
        modules.append(nn.Linear(input_dim, fnn_hidden_size_1))
        modules.append(nn.SiLU())
        
        # Hidden Layer 2: Linear -> Activation
        modules.append(nn.Linear(fnn_hidden_size_1, fnn_hidden_size_2))
        modules.append(nn.SiLU())
        
        # Output Layer: Linear
        modules.append(nn.Linear(fnn_hidden_size_2, output_dim))
        
        self.model = nn.Sequential(*modules)
        
        self.reset_c_parameters()

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def reset_c_parameters(self):
        mode = getattr(self, "init_c", "zero_last")
        mapper = self.model
        
        if mode == "zero_last":
            linear_layers = [m for m in mapper.modules() if isinstance(m, nn.Linear)]
            if linear_layers:
                last = linear_layers[-1]
                others = linear_layers[:-1]
                
                # Zero the last layer to ensure C starts as 0 (Identity via residual)
                nn.init.zeros_(last.weight)
                if last.bias is not None: nn.init.zeros_(last.bias)
                
                # Kaiming init for the common (hidden) layers
                for m in others:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None: nn.init.zeros_(m.bias)

        elif mode == "kaiming_uniform_m":
            for m in mapper.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                    if m.bias is not None: nn.init.zeros_(m.bias)
        elif mode == "zero_all":
            for m in mapper.modules():
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
                    if m.bias is not None: nn.init.zeros_(m.bias)

# =============================== NARA ================================

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass
class NARAConfig(PeftConfig):
    """
    NARA configuration.
    """

    # inits
    init_a: str = field(default="kaiming")
    init_b: str = field(default="zero")
    init_c: str = field(
        default="zero_last",
        metadata={"help": "FNN→C init: 'zero_last' zeros only the last Linear;"}
    )

    # LoRA sizes/targets
    r_ab: int = field(default=8)
    target_modules: Optional[Union[List[str], str]] = field(default=None)
    skip_layer_regex: Optional[str] = field(
        default=None,
        metadata={"help": "Regex for module names that should NOT receive NARA adapters."},
    )
    skip_layers: Optional[str] = field(
        default=None,
        metadata={"help": "Layer indexes/ranges to skip, e.g. 8-23,30."},
    )

    # core LoRA knobs
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.0) # Dropout only for LoRA leaves, removed from Mapper
    fan_in_fan_out: bool = field(default=False)
    bias: str = field(default="none")
    modules_to_save: Optional[List[str]] = field(default=None)
    init_lora_weights: bool = field(default=True)

    scale_ab: float = field(default=1.0)

    # train toggles
    train_a: bool = field(default=True)
    train_b: bool = field(default=True)
    train_mapper: bool = field(default=True)

    # mapper structure
    fnn_hidden_size_1: int = field(default=256)
    fnn_hidden_size_2: int = field(default=512)
    c_scale: float = field(default=1.0)



    # Format: {"qk_group": ["q_proj", "k_proj"], "vo_group": ["v_proj"]}
    # None = compatible with legacy mode (single global mapper shared by all)
    mapper_groups: Optional[Dict[str, List[str]]] = field(default=None)
    
    # --- INPUT MODE CONTROL ---
    # "nl" = Only Noise Level
    # "nd" = Only Noise Density
    # "both" = Noise Level + Noise Density (Concatenated)
    # "constant" = Learnable fixed C matrix (no noise input)
    input_mode: Literal["nl", "nd", "both", "constant"] = field(default="nl")

    # --- EMBEDDING CONTROL ---
    embedding_dim: int = field(default=64)
    # "fourier" (Default, Fixed) | "mlp" (Learnable) | "raw" (No embedding)
    embedding_type: Literal["fourier", "mlp", "raw"] = field(default="fourier")

    # --- DENSITY PARAMETERS ---
    density_radius: Optional[int] = field(default=None)
    direct_noise_level: bool = field(default=True)

    # def __post_init__(self):
    #     self.peft_type = "NARA" # Custom type string
    def __post_init__(self):
        self.peft_type = "NARA"  # Original code

        if self.mapper_groups is not None:
            
            # 1. Collect all patterns from mapper_groups
            all_group_patterns = []
            for patterns in self.mapper_groups.values():
                if not isinstance(patterns, (list, tuple)):
                    raise ValueError(f"mapper_groups values must be lists, got {type(patterns)}")
                all_group_patterns.extend(patterns)

            # 2. Check for internal overlaps (values must not overlap)
            # If list length != set length, there are duplicate elements
            if len(all_group_patterns) != len(set(all_group_patterns)):
                # Find duplicate elements for error reporting
                from collections import Counter
                duplicates = [item for item, count in Counter(all_group_patterns).items() if count > 1]
                raise ValueError(f"[NARA Config Error] mapper_groups patterns overlap (duplicates found): {duplicates}")

            # 3. Check consistency with target_modules (must cover all and not exceed -> set equality)
            # Prerequisite: target_modules must be a list for this static check
            # If target_modules is a regex string or None (default), skip this check or do partial check only
            if isinstance(self.target_modules, list):
                target_set = set(self.target_modules)
                group_set = set(all_group_patterns)

                # Check A: whether mapper_groups contains anything not in target_modules (must not exceed)
                extra_patterns = group_set - target_set
                if extra_patterns:
                    raise ValueError(
                        f"[NARA Config Error] mapper_groups contains patterns NOT in target_modules: {extra_patterns}. "
                        "Please ensure mapper_groups is a subset of target_modules."
                    )

                # Check B: whether target_modules has anything not covered by mapper_groups (must cover all)
                missing_patterns = target_set - group_set
                if missing_patterns:
                    raise ValueError(
                        f"[NARA Config Error] target_modules contains patterns NOT covered by mapper_groups: {missing_patterns}. "
                        "Every module in target_modules must belong to a group."
                    )
            
            elif isinstance(self.target_modules, str):
                # If target_modules is a regex string, static list comparison is difficult; just print a warning
                warnings.warn("target_modules is a string/regex, skipping strict coverage check for mapper_groups.")
        # [End of validation checks] --------------------------------------------



def _nara_parse_skip_layers(spec: Optional[str]) -> set[int]:
    """Parse a spec like "8-23,30" into a set of layer indexes."""
    if not spec:
        return set()
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            out.update(range(int(start), int(end) + 1))
        else:
            out.add(int(part))
    return out


def _nara_layer_index_from_key(key: str) -> Optional[int]:
    """Extract a transformer layer index from common module-name styles."""
    patterns = [
        r"(?:^|\.)layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)layer\.(\d+)(?:\.|$)",
        r"(?:^|\.)blocks\.(\d+)(?:\.|$)",
        r"(?:^|\.)block\.(\d+)(?:\.|$)",
        r"(?:^|\.)h\.(\d+)(?:\.|$)",
        r"(?:^|\.)decoder\.layers\.(\d+)(?:\.|$)",
        r"(?:^|\.)encoder\.layers\.(\d+)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, key)
        if m:
            return int(m.group(1))
    return None


def _nara_should_skip_key(key: str, lcfg) -> bool:
    """Return True when this target module should not receive a NARA adapter."""
    regex = getattr(lcfg, "skip_layer_regex", None) or os.environ.get("NARA_SKIP_LAYER_REGEX")
    if regex and re.search(regex, key):
        return True

    skip_layers = _nara_parse_skip_layers(os.environ.get("NARA_SKIP_LAYERS"))
    if not skip_layers:
        skip_layers = _nara_parse_skip_layers(getattr(lcfg, "skip_layers", None))
    if skip_layers:
        layer_idx = _nara_layer_index_from_key(key)
        return layer_idx in skip_layers

    return False

# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------

class NARAModel(nn.Module):
    def __init__(self, model, config, adapter_name: str):
        super().__init__()
        self.model = model
        self.forward = self.model.forward
        self.peft_config = config
        
        self.noise_level: Optional[torch.Tensor] = None
        self.noise_density: Optional[torch.Tensor] = None

        self.global_mapper = nn.ModuleDict({})
        self.embedding_layers = nn.ModuleDict({})
        self.constant_c = nn.ParameterDict({})  # For constant input_mode
        
        self.lora_layers = {} 
        
        self.c_matrix_cache: dict[str, List[torch.Tensor]] = {}
        self.c_cache_stale: dict[str, bool] = {}
        
        
        self.layer_group_mapping: dict[str, List[str]] = {}
        
        # --- Stage Control ---
        self.training_stage = 2  # Default to Stage 2 (ACB)
        # ---------------------

        self.add_adapter(adapter_name, self.peft_config[adapter_name])

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    def set_noise_level(self, noise_level: Optional[Union[float, torch.Tensor]] = None):
        """Legacy setter for just noise level."""
        self.set_context_state(noise_level, self.noise_density)

    def set_context_state(self, 
                          noise_level: Optional[Union[float, torch.Tensor]] = None,
                          noise_density: Optional[Union[float, torch.Tensor]] = None):
        """Unified setter for Level and Density. Triggers C pre-computation."""
        
        if noise_level is not None:
            if not torch.is_tensor(noise_level):
                noise_level = torch.tensor(noise_level, dtype=torch.float32)
        
        if noise_density is not None:
            if not torch.is_tensor(noise_density):
                noise_density = torch.tensor(noise_density, dtype=torch.float32)

        self.noise_level = noise_level
        self.noise_density = noise_density
        
        # ALWAYS recompute C matrices when context is set.
        for ad in self.peft_config:
            self.c_cache_stale[ad] = True
            
        for ad in self.peft_config:
            # Only recompute if the adapter is considered active (has layers)
            if ad in self.lora_layers and len(self.lora_layers[ad]) > 0:
                self._precompute_c_matrices(ad)

    def load_lora_only(self, ckpt_path: str, adapter_name: str = "default"):
            """
            Loads only the LoRA A and B matrices from a standard LoRA checkpoint.
            """
            print(f"\n[NARA] Loading LoRA A/B weights from {ckpt_path}...")
            
            if ckpt_path.endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    loaded_state_dict = load_file(ckpt_path, device="cpu")
                except ImportError:
                    raise ImportError("File ends with .safetensors but 'safetensors' library is not installed.")
            else:
                loaded_state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            filtered_dict = {}
            lora_keys_in_ckpt = 0
            loaded_keys_count = 0
            
            current_keys_map = {}
            for k in self.state_dict().keys():
                if f".{adapter_name}" in k and ("lora_A" in k or "lora_B" in k):
                    base_path = k.rsplit(f".{adapter_name}", 1)[0]
                    current_keys_map[base_path] = k
            
            for key, val in loaded_state_dict.items():
                if "lora_A" in key or "lora_B" in key:
                    lora_keys_in_ckpt += 1
                    key_norm = key
                    if key_norm.endswith(".weight"):
                        key_norm = key_norm[:-7]
                    
                    matched_target_key = None
                    if key_norm in current_keys_map:
                        matched_target_key = current_keys_map[key_norm]
                    elif key_norm.startswith("base_model.") and key_norm[11:] in current_keys_map:
                        matched_target_key = current_keys_map[key_norm[11:]]
                    elif key_norm.startswith("model.") and key_norm[6:] in current_keys_map:
                        matched_target_key = current_keys_map[key_norm[6:]]
                    elif f"base_model.{key_norm}" in current_keys_map:
                        matched_target_key = current_keys_map[f"base_model.{key_norm}"]

                    if matched_target_key:
                        current_shape = self.state_dict()[matched_target_key].shape
                        if val.shape == current_shape:
                            filtered_dict[matched_target_key] = val
                            loaded_keys_count += 1
                        elif val.shape == current_shape[::-1]:
                            filtered_dict[matched_target_key] = val.T
                            loaded_keys_count += 1
                            print(f"  - Transposed load for {key}")
                        else:
                            print(f"  - [Shape Mismatch] Skipping {key}. Ckpt: {val.shape}, Model: {current_shape}")

            if not filtered_dict:
                print("[NARA] WARNING: No matching LoRA keys found to load!")
                return

            missing, unexpected = self.load_state_dict(filtered_dict, strict=False)
            print("[NARA] Load Complete:")
            print(f"  - Total LoRA keys in Checkpoint: {lora_keys_in_ckpt}")
            print(f"  - Total Keys Loaded into Model:  {loaded_keys_count}")

    def set_training_stage(self, stage: int):
        """
        Switches between training stages.
        Stage 1: Train AB only (C acts as Identity).
        Stage 2: Train ACB (C is active and computed via Mapper).
        """
        if stage not in [1, 2]:
            raise ValueError("Stage must be 1 (AB only) or 2 (ACB).")
            
        print(f"[NARA] Switching to Training Stage {stage}")
        self.training_stage = stage
        
        for ad in self.peft_config:
            self.c_cache_stale[ad] = True
            
        if stage == 1:
            for mapper in self.global_mapper.values():
                mapper.requires_grad_(False)
            # Embedding layers (if learnable) should also be frozen
            for emb in self.embedding_layers.values():
                emb.requires_grad_(False)
        else:
            for mapper in self.global_mapper.values():
                mapper.requires_grad_(True)
            for emb in self.embedding_layers.values():
                emb.requires_grad_(True)
            
    # def _precompute_c_matrices(self, adapter_name: str):
    #     """
    #     Calculates C based on NL/ND (global context), then broadcasts it to all layers.
    #     """
    #     if adapter_name not in self.lora_layers or len(self.lora_layers[adapter_name]) == 0:
    #         return
            
    #     lcfg: NARAConfig = self.peft_config[adapter_name]
    #     mapper = self.global_mapper[adapter_name]
    #     emb_layers = self.embedding_layers
        
    #     any_p = next(mapper.parameters())
    #     dtype = any_p.dtype
    #     device = any_p.device
        
    #     R = lcfg.r_ab
    #     num_layers = len(self.lora_layers[adapter_name])
    #     Identity = torch.eye(R, dtype=dtype, device=device)

    #     # STAGE 1: AB Training Only (Bypass Mapper)
    #     if self.training_stage == 1:
    #         Ceff = Identity.contiguous()
    #         self.c_matrix_cache[adapter_name] = [Ceff] * num_layers
    #         self.c_cache_stale[adapter_name] = False
    #         return
        
    #     input_mode = lcfg.input_mode
    #     embeddings = []
        
    #     # Helper for embedding
    #     def get_emb(val, name_suffix):
    #         emb_mod = emb_layers[f"{adapter_name}_{name_suffix}_emb"]
    #         if val is None:
    #             raise ValueError(f"Context {name_suffix} required for input_mode={input_mode} but is None.")
    #         t_val = _as_tensor_like(val, any_p, dtype=dtype)
    #         return emb_mod(t_val) # Returns [1, dim]

    #     # 1. NL
    #     if input_mode in ["nl", "both"]:
    #         embeddings.append(get_emb(self.noise_level, "NL"))
        
    #     # 2. ND
    #     if input_mode in ["nd", "both"]:
    #         embeddings.append(get_emb(self.noise_density, "ND"))
            
    #     # Cat input
    #     fnn_input = torch.cat(embeddings, dim=-1) # [1, Total_Dim]
        
    #     # Pass through mapper
    #     Cn_flat = mapper(fnn_input)
        
    #     Cn = Cn_flat.view(R, R) # [R, R]
    #     c_scale = lcfg.c_scale
    #     Cn = (c_scale * Cn).to(dtype=dtype, device=device)

    #     # Ceff = Cn + I (Residual connection)
    #     Ceff = Cn + Identity
    #     Ceff = Ceff.contiguous()
        
    #     # Broadcast Ceff to all layers (Global Shared C)
    #     self.c_matrix_cache[adapter_name] = [Ceff] * num_layers
    #     self.c_cache_stale[adapter_name] = False
        
    def _precompute_c_matrices(self, adapter_name: str):
        """
        Calculates C based on NL/ND (global context), then broadcasts it to all layers.
        Modified to support Multiple Mappers (Groups) and Constant mode.
        """
        if adapter_name not in self.lora_layers or len(self.lora_layers[adapter_name]) == 0:
            return

        lcfg: NARAConfig = self.peft_config[adapter_name]
        emb_layers = self.embedding_layers

        # Get device/dtype from lora_layers (works for all modes including constant)
        first_layer = self.lora_layers[adapter_name][0]
        any_p = first_layer.lora_A[adapter_name]
        dtype = any_p.dtype
        device = any_p.device

        R = lcfg.r_ab
        num_layers = len(self.lora_layers[adapter_name])
        Identity = torch.eye(R, dtype=dtype, device=device)

        # STAGE 1: AB Training Only (Bypass Mapper)
        if self.training_stage == 1:
            Ceff = Identity.contiguous()
            self.c_matrix_cache[adapter_name] = [Ceff] * num_layers
            self.c_cache_stale[adapter_name] = False
            return

        # CONSTANT MODE: Use learnable parameter instead of Mapper
        if lcfg.input_mode == "constant":
            Cn = self.constant_c[adapter_name]
            # Match Identity to Cn's dtype/device to preserve gradient connection
            Identity_matched = torch.eye(R, dtype=Cn.dtype, device=Cn.device)
            Ceff = (lcfg.c_scale * Cn + Identity_matched).contiguous()
            self.c_matrix_cache[adapter_name] = [Ceff] * num_layers
            self.c_cache_stale[adapter_name] = False
            return

        # Normal modes: nl, nd, both - requires mapper
        relevant_keys = [k for k in self.global_mapper.keys() if k.startswith(adapter_name)]
        if not relevant_keys:
            raise RuntimeError(f"No mappers found for adapter {adapter_name}")

        input_mode = lcfg.input_mode
        embeddings = []

        # Helper for embedding
        def get_emb(val, name_suffix):
            emb_mod = emb_layers[f"{adapter_name}_{name_suffix}_emb"]
            if val is None:
                raise ValueError(f"Context {name_suffix} required for input_mode={input_mode} but is None.")
            t_val = _as_tensor_like(val, any_p, dtype=dtype)
            return emb_mod(t_val) # Returns [1, dim]

        # 1. NL
        if input_mode in ["nl", "both"]:
            embeddings.append(get_emb(self.noise_level, "NL"))

        # 2. ND
        if input_mode in ["nd", "both"]:
            embeddings.append(get_emb(self.noise_density, "ND"))

        # Cat input
        fnn_input = torch.cat(embeddings, dim=-1) # [1, Total_Dim]

        # --- [Core update start]: Compute C matrices for all groups ---
        group_c_map = {}

        # Iterate over all relevant Mappers
        for m_key in relevant_keys:
            mapper = self.global_mapper[m_key]

            # Parse group name
            if m_key == adapter_name:
                g_name = "DEFAULT_GROUP" # Legacy mode
            else:
                # Strip prefix "adapter_name_"
                g_name = m_key[len(adapter_name)+1:]

            Cn_flat = mapper(fnn_input)
            Cn = Cn_flat.view(R, R) # [R, R]
            c_scale = lcfg.c_scale
            Cn = (c_scale * Cn).to(dtype=dtype, device=device)

            # Ceff = Cn + I
            Ceff = (Cn + Identity).contiguous()
            group_c_map[g_name] = Ceff

        # import pdb; pdb.set_trace()
        # Assemble final list according to layer_group_mapping
        final_c_list = []
        current_mapping = self.layer_group_mapping[adapter_name]
        
        if len(current_mapping) != num_layers:
            # Defensive check: lengths should match in theory
            warnings.warn(f"Layer mapping length ({len(current_mapping)}) != num_layers ({num_layers}).")

        for g_name in current_mapping:
            if g_name in group_c_map:
                final_c_list.append(group_c_map[g_name])
            else:
                # If not found (possibly due to mapping mismatch when loading legacy weights), fall back to default
                if "DEFAULT_GROUP" in group_c_map:
                    final_c_list.append(group_c_map["DEFAULT_GROUP"])
                else:
                    raise RuntimeError(f"Group '{g_name}' required for layer but not computed.")

        self.c_matrix_cache[adapter_name] = final_c_list
        # --- [Core update end] ---
        self.c_cache_stale[adapter_name] = False

    def compute_c_smoothness_loss(
        self,
        adapter_name: str = "default",
        noise_level: Optional[Union[float, torch.Tensor]] = None,
        noise_density: Optional[Union[float, torch.Tensor]] = None,
        delta: float = 0.05,
    ) -> torch.Tensor:
        """
        Penalize abrupt changes in generated NARA cores along the noise trajectory.

        This is used as an optional training regularizer:
            ||C(lambda + delta) - C(lambda)||_F^2
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Unknown adapter: {adapter_name}")

        lcfg: NARAConfig = self.peft_config[adapter_name]
        if self.training_stage == 1 or lcfg.input_mode == "constant" or lcfg.input_mode not in ["nl", "both"]:
            ref = next(self.parameters())
            return ref.new_zeros(())

        relevant_keys = [k for k in self.global_mapper.keys() if k.startswith(adapter_name)]
        if not relevant_keys:
            ref = next(self.parameters())
            return ref.new_zeros(())

        mapper0 = self.global_mapper[relevant_keys[0]]
        any_p = next(mapper0.parameters())
        dtype = any_p.dtype
        device = any_p.device
        input_mode = lcfg.input_mode

        base_nl = self.noise_level if noise_level is None else noise_level
        if base_nl is None:
            return any_p.new_zeros(())
        base_nl = _as_tensor_like(base_nl, any_p, dtype=dtype).clamp(0.0, 1.0)
        next_nl = (base_nl + float(delta)).clamp(0.0, 1.0)
        if torch.allclose(base_nl, next_nl):
            next_nl = (base_nl - float(delta)).clamp(0.0, 1.0)
        if torch.allclose(base_nl, next_nl):
            return any_p.new_zeros(())

        base_nd = self.noise_density if noise_density is None else noise_density

        def build_input(nl_value: torch.Tensor) -> torch.Tensor:
            embeddings = []
            if input_mode in ["nl", "both"]:
                emb = self.embedding_layers[f"{adapter_name}_NL_emb"](nl_value)
                embeddings.append(emb)
            if input_mode in ["nd", "both"]:
                if base_nd is None:
                    raise ValueError("noise_density is required for NARA input_mode='both'.")
                nd_value = _as_tensor_like(base_nd, any_p, dtype=dtype)
                emb = self.embedding_layers[f"{adapter_name}_ND_emb"](nd_value)
                embeddings.append(emb)
            return torch.cat(embeddings, dim=-1)

        x0 = build_input(base_nl)
        x1 = build_input(next_nl)

        losses = []
        for key in relevant_keys:
            mapper = self.global_mapper[key]
            c0 = mapper(x0).float()
            c1 = mapper(x1).float()
            losses.append(F.mse_loss(c0, c1))

        return torch.stack(losses).mean().to(device=device)
        
    def add_adapter(self, adapter_name, config: Optional[NARAConfig] = None):
        if config is not None:
            model_config = (
                self.model.config.to_dict()
                if hasattr(self.model.config, "to_dict")
                else self.model.config
            )
            config = self._prepare_lora_config(config, model_config)
            self.peft_config[adapter_name] = config
 
        if adapter_name not in self.lora_layers:
            self.lora_layers[adapter_name] = [] 
            
        self.c_matrix_cache[adapter_name] = []
        self.c_cache_stale[adapter_name] = True
        
        self.layer_group_mapping[adapter_name] = []
        
        self._find_and_replace(adapter_name)

        # ----------------------------------------------------------------
        # Instantiate Global Mapper and Embeddings
        # ----------------------------------------------------------------
        lcfg: NARAConfig = self.peft_config[adapter_name]
        r_ab = lcfg.r_ab
        emb_dim = lcfg.embedding_dim
        input_mode = lcfg.input_mode
        emb_type = lcfg.embedding_type # "fourier", "mlp", "raw"
        
        # --- Helper to create embedding module ---
        def create_emb_module():
            if emb_type == "fourier":
                # Explicitly set scale=16.0 here for proper [0,1] input coverage
                return GaussianFourierProjection(embed_dim=emb_dim, scale=16.0)
            elif emb_type == "mlp":
                return MLPEmbedding(embed_dim=emb_dim)
            elif emb_type == "raw":
                return RawEmbedding()
            else:
                raise ValueError(f"Unknown embedding_type: {emb_type}")

        # --- Calculate total input dimension ---
        # 1. Determine "Unit Dimension" (dim of single component)
        if emb_type == "raw":
            unit_dim = 1
        else:
            unit_dim = emb_dim

        # 2. Determine Total Dim based on inputs
        total_input_dim = 0
        if input_mode == "nl":
            total_input_dim = unit_dim
        elif input_mode == "nd":
            total_input_dim = unit_dim
        elif input_mode == "both":
            total_input_dim = unit_dim * 2

        # # 1. Mapper
        # mapper = NARAMapper(
        #     r_ab=r_ab,
        #     input_dim=total_input_dim,
        #     fnn_hidden_size_1=lcfg.fnn_hidden_size_1,
        #     fnn_hidden_size_2=lcfg.fnn_hidden_size_2,
        #     init_c=lcfg.init_c
        # )
        # self.global_mapper.update(nn.ModuleDict({adapter_name: mapper}))

        # === CONSTANT MODE: Create learnable C parameter instead of Mapper ===
        if input_mode == "constant":
            self.constant_c.update(nn.ParameterDict({
                adapter_name: nn.Parameter(torch.zeros(r_ab, r_ab))
            }))
            # Skip mapper and embedding creation for constant mode
        else:
            # [Replaced with the following code] -------------------------------------------------
            # 1. Mapper (supports multiple groups)
            groups_to_create = []
            if lcfg.mapper_groups is None:
                # Legacy mode: use adapter_name directly as key
                groups_to_create.append(adapter_name)
            else:
                # New mode: key is adapter_name + "_" + group_name
                for g_name in lcfg.mapper_groups.keys():
                    groups_to_create.append(f"{adapter_name}_{g_name}")

            for map_key in groups_to_create:
                mapper = NARAMapper(
                    r_ab=r_ab,
                    input_dim=total_input_dim,
                    fnn_hidden_size_1=lcfg.fnn_hidden_size_1,
                    fnn_hidden_size_2=lcfg.fnn_hidden_size_2,
                    init_c=lcfg.init_c
                )
                self.global_mapper.update(nn.ModuleDict({map_key: mapper}))

            # ------------------------------------------------------------------
            # 2. Embeddings
            if input_mode in ["nl", "both"]:
                self.embedding_layers.update(nn.ModuleDict({
                    f"{adapter_name}_NL_emb": create_emb_module()
                }))

            if input_mode in ["nd", "both"]:
                self.embedding_layers.update(nn.ModuleDict({
                    f"{adapter_name}_ND_emb": create_emb_module()
                }))

        if len(self.peft_config) > 1 and self.peft_config[adapter_name].bias != "none":
            raise ValueError("NARAModel supports only 1 adapter with bias.")
 
        mark_only_ContextLoRA_as_trainable(self.model, self.peft_config[adapter_name].bias, config)
 
        if self.peft_config[adapter_name].inference_mode:
            _freeze_adapter(self.model, adapter_name)

    def _find_and_replace(self, adapter_name: str):
        lcfg: NARAConfig = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError("`bitsandbytes` is required for 8-bit quantization.")
 
        is_target_modules_in_base_model = False
        
        # Tracking global index for the C matrix cache
        layer_global_index = 0
        
        kwargs = {
            "r_ab": lcfg.r_ab,
            "lora_alpha": lcfg.lora_alpha,
            "lora_dropout": lcfg.lora_dropout,
            "fan_in_fan_out": lcfg.fan_in_fan_out,
            "init_lora_weights": lcfg.init_lora_weights,
            "scale_ab": lcfg.scale_ab,
            "init_a": lcfg.init_a,
            "init_b": lcfg.init_b,
            "init_c": lcfg.init_c,
            "train_a": lcfg.train_a,
            "train_b": lcfg.train_b,
            "c_scale": lcfg.c_scale,
        }
 
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(lcfg.target_modules, str):
                target_module_found = re.fullmatch(lcfg.target_modules, key)
            else:
                target_module_found = any(key.endswith(tk) for tk in lcfg.target_modules)
 
            if not target_module_found:
                continue

            if _nara_should_skip_key(key, lcfg):
                if os.environ.get("NARA_DEBUG_SKIP_LAYERS"):
                    print(f"[NARA skip] {key}")
                continue

            if os.environ.get("NARA_DEBUG_TARGETS"):
                print(f"[NARA target] {key}")

 
            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.model, key)
            bias = getattr(target, "bias", None) is not None
            # Determine which Group the current layer belongs to
            current_group_name = "DEFAULT_GROUP"
            
            if lcfg.mapper_groups is not None:
                found_group = None
                for g_name, patterns in lcfg.mapper_groups.items():
                    for pat in patterns:
                        # Use regex search or suffix matching
                        if key.endswith(pat) or re.search(pat, key):
                            found_group = g_name
                            break
                    if found_group:
                        break
                
                if found_group:
                    current_group_name = found_group
                else:
                    # If the module is in target_modules but not defined in any group, assign to the first group by default
                    if len(lcfg.mapper_groups) > 0:
                        current_group_name = list(lcfg.mapper_groups.keys())[0]

            self.layer_group_mapping[adapter_name].append(current_group_name)
            if isinstance(target, NARALayer):
                warnings.warn(f"Re-initializing NARALayer for {key}. This is usually unexpected.")
                continue
            else:
                if isinstance(target, nn.Linear):
                    in_features, out_features = target.in_features, target.out_features
                    if kwargs["fan_in_fan_out"]:
                        warnings.warn("fan_in_fan_out=True with torch.nn.Linear; forcing False.")
                        kwargs["fan_in_fan_out"] = (lcfg.fan_in_fan_out) = False
                elif isinstance(target, Conv1D):
                    in_features, out_features = (
                        target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                    )
                    if not kwargs["fan_in_fan_out"]:
                        warnings.warn("fan_in_fan_out=False with Conv1D; forcing True.")
                        kwargs["fan_in_fan_out"] = (lcfg.fan_in_fan_out) = True
                else:
                    continue
 
                new_module = NARALinear(
                    adapter_name,
                    in_features,
                    out_features,
                    bias=bias,
                    parent_model=self,
                    layer_global_index=layer_global_index,
                    **kwargs,
                )
                self._replace_module(parent, target_name, new_module, target)
                
                self.lora_layers[adapter_name].append(new_module)
                layer_global_index += 1
 
        if not is_target_modules_in_base_model:
            raise ValueError(f"Target modules {lcfg.target_modules} not found.")
 
    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias") and old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)
        
        # Ensure new module parameters are on the correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
        
        # Ensure Global Mapper/Embeddings are also on the correct device
        ad = new_module.active_adapter
        if ad in self.global_mapper:
             self.global_mapper[ad].to(old_module.weight.device)
        for name, module in self.embedding_layers.items():
            if name.startswith(ad):
                 module.to(old_module.weight.device)
        # Move constant_c to correct device if it exists
        if ad in self.constant_c:
            self.constant_c[ad].data = self.constant_c[ad].data.to(old_module.weight.device)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
            config_dict[key] = config
        return config_dict
      
    @staticmethod
    def _prepare_lora_config(peft_config: NARAConfig, model_config):
        if peft_config.target_modules is None:
            mt = model_config["model_type"]
            if mt in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[mt]
        if peft_config.inference_mode:
            peft_config.merge_weights = True
        return peft_config
 
def mark_only_ContextLoRA_as_trainable(
    model: nn.Module, bias: str = "none", config: Optional[NARAConfig] = None
) -> None:
    for n, p in model.named_parameters():
        p.requires_grad = False
    if bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, NARALayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
      
    if config:
        if config.train_a:
            for n, p in model.named_parameters():
                if "lora_A" in n: p.requires_grad = True
        if config.train_b:
            for n, p in model.named_parameters():
                if "lora_B" in n: p.requires_grad = True
        if config.train_mapper:
            for n, p in model.named_parameters():
                # Enable grads for global_mapper, embedding_layers, AND constant_c
                if "global_mapper" in n or "embedding_layers" in n or "constant_c" in n:
                    p.requires_grad = True
 
# ---------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------
class NARALayer:
    """
    Base class for NARA layers. Stores global index to access the shared C cache.
    """
    def __init__(self, in_features: int, out_features: int, layer_global_index: int):
        self.r_ab = {}
        self.lora_alpha = {}
        self.scaling_ab = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ParameterDict({})
        self.lora_B = nn.ParameterDict({})
        
        # Index to retrieve pre-computed C matrix from the model cache
        self.layer_global_index = layer_global_index 

        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.init_a = None
        self.init_b = None
        self.init_c = None
        self.c_scale = {}
          
        
    def update_layer(
        self,
        adapter_name: str,
        r_ab: int,
        lora_alpha: float,
        lora_dropout: float,
        init_lora_weights: bool,
        scale_ab: float,
        init_a: str,
        init_b: str,
        init_c: str,
        c_scale: float,
    ):
        self.r_ab[adapter_name] = int(r_ab)
        self.init_a = init_a
        self.init_b = init_b
        self.init_c = init_c 
        self.lora_alpha[adapter_name] = float(lora_alpha)
        self.c_scale[adapter_name] = float(c_scale)
 
        self.lora_dropout.update(
            nn.ModuleDict(
                {adapter_name: nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()}
            )
        )
 
        if r_ab > 0:
            self.lora_A.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(r_ab, self.in_features))}))
            self.lora_B.update(nn.ParameterDict({adapter_name: nn.Parameter(torch.randn(self.out_features, r_ab))}))
            self.scaling_ab[adapter_name] = float(scale_ab)
 
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)
 
    def reset_lora_parameters(self, adapter_name: str):
        init_mapping = {"kaiming": nn.init.kaiming_uniform_, "zero": nn.init.zeros_}
        init_kwargs = {"kaiming": {"a": math.sqrt(5)}, "zero": {}}
 
        if self.init_a not in init_mapping or self.init_b not in init_mapping:
            raise ValueError(f"Invalid init type.")
 
        if adapter_name in self.lora_A and self.r_ab.get(adapter_name, 0) > 0:
            init_mapping[self.init_a](self.lora_A[adapter_name], **init_kwargs[self.init_a])
            init_mapping[self.init_b](self.lora_B[adapter_name], **init_kwargs[self.init_b])
        
# ---------------------------------------------------------------------
# Wrapped Linear module
# ---------------------------------------------------------------------
class NARALinear(nn.Linear, NARALayer):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        # NARA specific context
        layer_global_index: int,
        # LoRA params
        r_ab: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,
        scale_ab: float = 1.0,
        init_a: str = "kaiming",
        init_b: str = "zero",
        init_c: str = "zero_last",
        train_a: bool = True,
        train_b: bool = True,
        parent_model: Optional[NARAModel] = None,
        c_scale: float = 1.0,
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # Initialize NARALayer first
        NARALayer.__init__(self, in_features=in_features, out_features=out_features, 
                                layer_global_index=layer_global_index)
 
        self.weight.requires_grad = False
        self.train_a = train_a
        self.train_b = train_b
        self.fan_in_fan_out = fan_in_fan_out
        self._parent_ref = weakref.ref(parent_model) if parent_model is not None else None
 
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
 
        nn.Linear.reset_parameters(self)
 
        self.update_layer(
            adapter_name,
            r_ab,
            lora_alpha,
            lora_dropout,
            init_lora_weights,
            scale_ab,
            init_a,
            init_b,
            init_c,
            c_scale,
        )
        self.active_adapter = adapter_name
 
    @property
    def noise_level(self) -> Optional[torch.Tensor]:
        parent = self._parent_ref() if self._parent_ref is not None else None
        return getattr(parent, "noise_level", None)
      
    @property
    def noise_density(self) -> Optional[torch.Tensor]:
        parent = self._parent_ref() if self._parent_ref is not None else None
        return getattr(parent, "noise_density", None)
 
    def forward(self, x: torch.Tensor):
        prev_dtype = x.dtype
        base_w = transpose(self.weight, self.fan_in_fan_out)
        # base_out = F.linear(x, base_w, bias=self.bias)
        # 修改后 🌟（将 x 的 dtype 强制转换为与 base_w 一致）
        base_out = F.linear(x.to(base_w.dtype), base_w, bias=self.bias)
 
        if self.disable_adapters or (self.active_adapter not in self.lora_A):
            return base_out.to(prev_dtype)
        if self.merged or self.r_ab.get(self.active_adapter, 0) <= 0:
            return base_out.to(prev_dtype)

        ad = self.active_adapter
        parent = self._parent_ref() 
        if parent is None:
            raise RuntimeError("Parent NARAModel reference lost.")
        
        # Check for cache staleness and recompute if necessary 
        if parent.c_cache_stale.get(ad, False):
            # This should only happen if set_context_state wasn't called defensively, 
            # but we ensure computation happens here if needed.
            parent._precompute_c_matrices(ad) 
        
        # ----------------------------------------------------
        # Retrieve Pre-computed Ceff matrix (R x R)
        # ----------------------------------------------------
        c_matrix_list = parent.c_matrix_cache.get(ad)
        if not c_matrix_list or self.layer_global_index >= len(c_matrix_list):
             # Fallback: if cache is empty or index is out of bounds, raise error
            raise RuntimeError(f"C matrix cache for adapter '{ad}' is invalid or missing entry for global index {self.layer_global_index}.")
        
        # Ceff is [R, R]
        Ceff = c_matrix_list[self.layer_global_index]
        
        # ----------------------------------------------------
        # Compute Delta using the cached Ceff (R x R)
        # ----------------------------------------------------
        A = self.lora_A[ad]
        B = self.lora_B[ad]
        scale = self.scaling_ab[ad]
 
        x_lora = self.lora_dropout[ad](x).to(A.dtype)
        h = F.linear(x_lora, A) # [B, R]
 
        # Apply Ceff: h @ Ceff. Since Ceff is [R, R], this is standard matrix multiplication.
        # h is [B, R]. Ceff is [R, R]. Result h_t is [B, R].
        h_t = torch.matmul(h, Ceff)
        
        # delta = h_t @ B^T
        delta = F.linear(h_t, B) * scale # [B, Out_Features]
 
        return (base_out + delta).to(prev_dtype)
