import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel  # type: ignore


import math
from peft import PeftModel

from config import  FINETUNING_TYPE
from utils import forward_with_noise_level,calculate_global_mask_density


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def _noise_level_excluding_prompt(x: torch.Tensor,
                                 prompt_index: torch.Tensor,
                                 mask_id: int) -> torch.Tensor:
    
    non_prompt = ~prompt_index                          # bool [B, L]

    non_prompt_masked = (x == mask_id) & non_prompt     # bool [B, L]
    
    num = non_prompt_masked.float().sum(dim=1, keepdim=True)  # [B, 1]
    den = non_prompt.float().sum(dim=1, keepdim=True)         # [B, 1]
    
    zero = den.new_zeros(den.shape)
    noise_level = torch.where(den > 0, num / den, zero)       # [B, 1]

    return noise_level

def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens

# @torch.no_grad()
# def generate(
#     model,
#     finetuning_type:FINETUNING_TYPE,
#     direct_noise:bool,
#     prompt,
#     steps=128,
#     gen_length=128,
#     block_length=128,
#     temperature=0.0,
#     cfg_scale=0.0,
#     remasking="low_confidence",
#     mask_id=126336,
#     is_main_process: bool = True,
#     random_noise: bool = False,
# ):
#     """
#     Args:
#         model: Mask predictor.
#         prompt: A tensor of shape (1, L).
#         steps: Sampling steps, less than or equal to gen_length.
#         gen_length: Generated answer length.
#         block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
#         temperature: Categorical distribution sampling temperature.
#         cfg_scale: Unsupervised classifier-free guidance scale.
#         remasking: Remasking strategy. 'low_confidence' or 'random'.
#         mask_id: The token id of [MASK] is 126336.
#     """
#     x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
#         model.device
#     )
#     x[:, : prompt.shape[1]] = prompt.clone()

#     prompt_index = x != mask_id

#     assert gen_length % block_length == 0
#     num_blocks = gen_length // block_length

#     assert steps % num_blocks == 0
#     steps = steps // num_blocks
#     steps_bar = tqdm(
#         total=steps,
#         initial=0,
#         desc="Steps",
#         leave=False,
#         disable=not is_main_process,
#     )
#     for num_block in range(num_blocks):
#         block_mask_index = (
#             x[
#                 :,
#                 prompt.shape[1] + num_block * block_length : prompt.shape[1]
#                 + (num_block + 1) * block_length :,
#             ]
#             == mask_id
#         )
#         num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
#         steps_bar.reset()
#         for i in range(steps):
#             import pdb; pdb.set_trace()
#             mask_index = (x == mask_id)
#             if (finetuning_type is not None) and (finetuning_type in (FINETUNING_TYPE.TLORA,FINETUNING_TYPE.NORA,FINETUNING_TYPE.TNORA)):
#                 if cfg_scale > 0.0:
#                     un_x = x.clone()
#                     if not direct_noise:
#                         guided_masked_indices_float = (x == mask_id).float()
#                         guided_noise_level = guided_masked_indices_float.mean(
#                             dim=1, keepdim=True
#                         )
                        
#                         logits_guided = forward_with_noise_level(
#                             model, x, guided_noise_level,random_noise
#                         )
#                         un_x[prompt_index] = mask_id
#                         unguided_masked_indices_float = (un_x == mask_id).float()
#                         unguided_noise_level = unguided_masked_indices_float.mean(
#                             dim=1, keepdim=True
#                         )
#                         logits_unguided = forward_with_noise_level(
#                             model, un_x, unguided_noise_level,random_noise
#                         )
#                         logits = logits_unguided + (cfg_scale + 1) * (
#                             logits_guided - logits_unguided
#                         )
#                     else:
#                         guided_noise_level=_noise_level_excluding_prompt(x, prompt_index, mask_id)
#                         logits_guided = forward_with_noise_level(
#                             model, x, guided_noise_level,random_noise
#                         )
#                         logits_unguided = forward_with_noise_level(
#                             model, un_x, guided_noise_level,random_noise
#                         )
#                         logits = logits_unguided + (cfg_scale + 1) * (
#                             logits_guided - logits_unguided
#                         )

#                 else:
#                     if not direct_noise:
#                         masked_indices_float = (x == mask_id).float()
#                         noise_level = masked_indices_float.mean(dim=1, keepdim=True)
#                         logits = forward_with_noise_level(model, x, noise_level,random_noise)
#                     else:
#                         noise_level=_noise_level_excluding_prompt(x, prompt_index, mask_id)
#                         logits = forward_with_noise_level(model, x, noise_level,random_noise)
#             else:
#                 if cfg_scale > 0.0:
#                     un_x = x.clone()
#                     un_x[prompt_index] = mask_id
#                     x_ = torch.cat([x, un_x], dim=0)
#                     logits = model(x_).logits
#                     logits, un_logits = torch.chunk(logits, 2, dim=0)
#                     logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
#                 else:
#                     logits = model(x).logits

#             logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
#             x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

#             if remasking == "low_confidence":
#                 p = F.softmax(logits, dim=-1)
#                 x0_p = torch.squeeze(
#                     torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
#                 )  # b, l
#             elif remasking == "random":
#                 x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
#             else:
#                 raise NotImplementedError(remasking)

#             x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

#             x0 = torch.where(mask_index, x0, x)
#             confidence = torch.where(mask_index, x0_p, -np.inf)

#             transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
#             for j in range(confidence.shape[0]):
#                 _, select_index = torch.topk(
#                     confidence[j], k=int(num_transfer_tokens[j, i])
#                 )
#                 transfer_index[j, select_index] = True
#             x[transfer_index] = x0[transfer_index]
#             steps_bar.update(1)
#             steps_bar.set_postfix({"Blocks": f"{num_block + 1}/{num_blocks}"})
#     return x

@torch.no_grad()
def generate_for_varlenlora(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    is_main_process: bool = True,
    whole_length: bool = False,
):
    """
    Specialized generation function for VarLenLoRA.
    Optimized with 'Smart Merge': Merges adapter weights for the specific target length
    and keeps them merged across calls to accelerate sequential generation.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    steps_bar = tqdm(
        total=steps,
        initial=0,
        desc="Steps (VarLen)",
        leave=False,
        disable=not is_main_process,
    )

    # --- VarLenLoRA Smart Merge Setup ---
    real_model = model.module if hasattr(model, "module") else model
    adapter_name = "default"

    # Check the last length we merged for. 
    # If None, it implies the model is currently unmerged (or fresh).
    last_merged_len = getattr(real_model, "_current_merged_length", None)
    target_length = gen_length+prompt.shape[1] if whole_length else gen_length
    # Only change state if the requested length differs from the cached merged state
    if last_merged_len != target_length:
        # 1. Unmerge if previously merged with a different length
        if last_merged_len is not None:
            if hasattr(real_model, "unmerge_adapter"):
                real_model.unmerge_adapter(adapter_name)
                real_model._current_merged_length = None # Update state
        
        # 2. Merge with the new target length
        if hasattr(real_model, "merge_adapter"):
            real_model.merge_adapter(adapter_name=adapter_name, target_length=target_length)
            # Cache the state so we skip this next time
            real_model._current_merged_length = target_length
        
        elif hasattr(real_model, "set_target_length"):
            # Fallback for models that support length setting but not merging
            real_model.set_target_length(target_length)

    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        steps_bar.reset()

        for i in range(steps):
            mask_index = (x == mask_id)

            # Standard forward pass (Adapter weights are baked in via merge)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], k=int(num_transfer_tokens[j, i])
                )
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            steps_bar.update(1)
            steps_bar.set_postfix({"Blocks": f"{num_block + 1}/{num_blocks}"})

    return x

@torch.no_grad()
def generate(
    model,
    tokenizer,
    finetuning_type: FINETUNING_TYPE,
    direct_noise: bool,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    is_main_process: bool = True,
    random_noise: bool = False,
    whole_length: bool = False,
    till_eos: bool = False,
    till_current_eos: bool = False,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK].
    """

    # --- DISPATCHER: REDIRECT VARLENLORA ---
    if finetuning_type == getattr(FINETUNING_TYPE, "VARLENLORA", "VARLENLORA"):
        return generate_for_varlenlora(
            model=model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=mask_id,
            is_main_process=is_main_process,
            whole_length=whole_length,
        )
    # ---------------------------------------

    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    steps_bar = tqdm(
        total=steps,
        initial=0,
        desc="Steps",
        leave=False,
        disable=not is_main_process,
    )
    
    # ==========================================
    #        CONFIGURATION SETUP
    # ==========================================
    
    nora_input_mode = None
    nora_density_radius = None
    clora_input_components = []
    clora_density_radius = None

    # --- 1. NORA Configuration Setup (Original Logic) ---
    if finetuning_type == FINETUNING_TYPE.NORA:
        real_model = model.module if hasattr(model, "module") else model
        peft_cfg = getattr(real_model, "peft_config", {}).get('default', None)
        
        if peft_cfg is None:
            raise ValueError("Finetuning type is NORA, but no default adapter config found.")
        
        if not hasattr(peft_cfg, "input_mode"):
            raise ValueError("NORA adapter config is missing 'input_mode'.")
        
        nora_input_mode = peft_cfg.input_mode
        
        if nora_input_mode in ["noise_density", "both"]:
            if not hasattr(peft_cfg, "density_radius") or peft_cfg.density_radius is None:
                raise ValueError(f"NORA input_mode is '{nora_input_mode}' but 'density_radius' is missing or None.")
            nora_density_radius = peft_cfg.density_radius

    # --- 2. CLoRA Configuration Setup (New Logic) ---
    elif finetuning_type == FINETUNING_TYPE.CLORA:
        real_model = model.module if hasattr(model, "module") else model
        peft_cfg = getattr(real_model, "peft_config", {}).get('default', None)
        
        if peft_cfg is not None:
            # Retrieve CLoRA specific attributes
            clora_input_components = getattr(peft_cfg, "input_components", [])
            clora_density_radius = getattr(peft_cfg, "density_radius", None)
            
            # Validation
            if "nd" in clora_input_components and clora_density_radius is None:
                raise ValueError("CLORA config uses 'nd' (Noise Density) but 'density_radius' is missing/None.")
    elif finetuning_type == FINETUNING_TYPE.NARA:
        real_model = model.module if hasattr(model, "module") else model
        peft_cfg = getattr(real_model, "peft_config", {}).get('default', None)
        
        if peft_cfg is not None:
            # Retrieve CLoRA specific attributes
            nara_input_components = getattr(peft_cfg, "input_mode", None)
            nara_density_radius = getattr(peft_cfg, "density_radius", None)
            
            # Validation
            if nara_input_components in ("nd","both") and nara_density_radius is None:
                raise ValueError("NARA config uses 'nd' (Noise Density) but 'density_radius' is missing/None.")
    # ==========================================
    #        GENERATION LOOP
    # ==========================================
    stop_after_current_block = False
    
    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        steps_bar.reset()
        
        for i in range(steps):
            mask_index = (x == mask_id)

            # --- BRANCH 1: NORA (Original Logic) ---
            if finetuning_type == FINETUNING_TYPE.NORA:
                
                def get_nora_args(batch_mask_index, raw_noise_level):
                    """NORA-specific argument resolver based on input_mode string"""
                    args = {"randomize_noise": random_noise}
                    
                    # Calculate Density if needed
                    nd = None
                    if nora_input_mode in ["noise_density", "both"]:
                        nd = calculate_global_mask_density(batch_mask_index, r=nora_density_radius)
                    
                    # Assign based on mode
                    if nora_input_mode == "noise_level":
                        args["noise_level"] = raw_noise_level
                        args["noise_density"] = None
                    elif nora_input_mode == "noise_density":
                        args["noise_level"] = None
                        args["noise_density"] = nd
                    elif nora_input_mode == "both":
                        args["noise_level"] = raw_noise_level
                        args["noise_density"] = nd
                        
                    return args

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    if not direct_noise:
                        # Guided
                        guided_masked_indices_float = (x == mask_id).float()
                        guided_noise_level = guided_masked_indices_float.mean(dim=1, keepdim=True)
                        fwd_args = get_nora_args(mask_index, guided_noise_level)
                        logits_guided = forward_with_noise_level(model, x, **fwd_args)
                        
                        # Unguided
                        un_x[prompt_index] = mask_id
                        unguided_masked_indices_float = (un_x == mask_id).float()
                        unguided_noise_level = unguided_masked_indices_float.mean(dim=1, keepdim=True)
                        un_mask_index = (un_x == mask_id)
                        fwd_args_un = get_nora_args(un_mask_index, unguided_noise_level)
                        logits_unguided = forward_with_noise_level(model, un_x, **fwd_args_un)

                        logits = logits_unguided + (cfg_scale + 1) * (logits_guided - logits_unguided)
                    else:
                        # Direct noise logic
                        guided_noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id)
                        fwd_args = get_nora_args(mask_index, guided_noise_level)
                        logits_guided = forward_with_noise_level(model, x, **fwd_args)
                        logits_unguided = forward_with_noise_level(model, un_x, **fwd_args)
                        logits = logits_unguided + (cfg_scale + 1) * (logits_guided - logits_unguided)
                else:
                    if not direct_noise:
                        masked_indices_float = (x == mask_id).float()
                        noise_level = masked_indices_float.mean(dim=1, keepdim=True)
                        fwd_args = get_nora_args(mask_index, noise_level)
                        logits = forward_with_noise_level(model, x, **fwd_args)
                    else:
                        noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id)
                        fwd_args = get_nora_args(mask_index, noise_level)
                        logits = forward_with_noise_level(model, x, **fwd_args)

            # --- BRANCH 2: CLORA (New Logic) ---
            elif finetuning_type == FINETUNING_TYPE.CLORA:
                
                def get_clora_args(batch_mask_index, raw_noise_level):
                    """CLoRA-specific argument resolver based on input_components list"""
                    args = {"randomize_noise": random_noise}
                    
                    # 1. Noise Level (nl)
                    if "nl" in clora_input_components:
                        args["noise_level"] = raw_noise_level
                    else:
                        args["noise_level"] = None

                    # 2. Noise Density (nd)
                    if "nd" in clora_input_components:
                        nd = calculate_global_mask_density(batch_mask_index, r=clora_density_radius)
                        args["noise_density"] = nd
                    else:
                        args["noise_density"] = None
                        
                    return args

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    if not direct_noise:
                        # Guided
                        guided_masked_indices_float = (x == mask_id).float()
                        guided_noise_level = guided_masked_indices_float.mean(dim=1, keepdim=True)
                        fwd_args = get_clora_args(mask_index, guided_noise_level)
                        logits_guided = forward_with_noise_level(model, x, **fwd_args)
                        
                        # Unguided
                        un_x[prompt_index] = mask_id
                        unguided_masked_indices_float = (un_x == mask_id).float()
                        unguided_noise_level = unguided_masked_indices_float.mean(dim=1, keepdim=True)
                        un_mask_index = (un_x == mask_id)
                        fwd_args_un = get_clora_args(un_mask_index, unguided_noise_level)
                        logits_unguided = forward_with_noise_level(model, un_x, **fwd_args_un)

                        logits = logits_unguided + (cfg_scale + 1) * (logits_guided - logits_unguided)
                    else:
                        # Direct noise logic
                        guided_noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id)
                        fwd_args = get_clora_args(mask_index, guided_noise_level)
                        logits_guided = forward_with_noise_level(model, x, **fwd_args)
                        logits_unguided = forward_with_noise_level(model, un_x, **fwd_args)
                        logits = logits_unguided + (cfg_scale + 1) * (logits_guided - logits_unguided)
                else:
                    if not direct_noise:
                        masked_indices_float = (x == mask_id).float()
                        noise_level = masked_indices_float.mean(dim=1, keepdim=True)
                        fwd_args = get_clora_args(mask_index, noise_level)
                        logits = forward_with_noise_level(model, x, **fwd_args)
                    else:
                        noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id)
                        fwd_args = get_clora_args(mask_index, noise_level)
                        logits = forward_with_noise_level(model, x, **fwd_args)
            # --- BRANCH 3: NARA (New Logic) ---
            elif finetuning_type == FINETUNING_TYPE.NARA:
                
                def get_nara_args(batch_mask_index, raw_noise_level):
                    """CLoRA-specific argument resolver based on input_components list"""
                    args = {"randomize_noise": random_noise}
                    
                    # 1. Noise Level (nl)
                    if nara_input_components in ("nl","both"):
                        args["noise_level"] = raw_noise_level
                    else:
                        args["noise_level"] = None

                    # 2. Noise Density (nd)
                    if nara_input_components in ("nd","both"):
                        nd = calculate_global_mask_density(batch_mask_index, r=nara_density_radius)
                        args["noise_density"] = nd
                    else:
                        args["noise_density"] = None
                        
                    return args

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    if not direct_noise:
                        # Guided
                        guided_masked_indices_float = (x == mask_id).float()
                        guided_noise_level = guided_masked_indices_float.mean(dim=1, keepdim=True)
                        fwd_args = get_nara_args(mask_index, guided_noise_level)
                        logits_guided = forward_with_noise_level(model, x, **fwd_args)
                        
                        # Unguided
                        un_x[prompt_index] = mask_id
                        unguided_masked_indices_float = (un_x == mask_id).float()
                        unguided_noise_level = unguided_masked_indices_float.mean(dim=1, keepdim=True)
                        un_mask_index = (un_x == mask_id)
                        fwd_args_un = get_nara_args(un_mask_index, unguided_noise_level)
                        logits_unguided = forward_with_noise_level(model, un_x, **fwd_args_un)

                        logits = logits_unguided + (cfg_scale + 1) * (logits_guided - logits_unguided)
                    else:
                        # Direct noise logic
                        guided_noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id)
                        fwd_args = get_nara_args(mask_index, guided_noise_level)
                        logits_guided = forward_with_noise_level(model, x, **fwd_args)
                        logits_unguided = forward_with_noise_level(model, un_x, **fwd_args)
                        logits = logits_unguided + (cfg_scale + 1) * (logits_guided - logits_unguided)
                else:
                    if not direct_noise:
                        masked_indices_float = (x == mask_id).float()
                        noise_level = masked_indices_float.mean(dim=1, keepdim=True)
                        fwd_args = get_nara_args(mask_index, noise_level)
                        logits = forward_with_noise_level(model, x, **fwd_args)
                    else:
                        noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id)
                        fwd_args = get_nara_args(mask_index, noise_level)
                        logits = forward_with_noise_level(model, x, **fwd_args)
            # --- BRANCH 3: TLORA / TNORA (Original Logic) ---
            elif finetuning_type in (FINETUNING_TYPE.TLORA, FINETUNING_TYPE.TNORA):
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    if not direct_noise:
                        guided_masked_indices_float = (x == mask_id).float()
                        guided_noise_level = guided_masked_indices_float.mean(
                            dim=1, keepdim=True
                        )
                        
                        logits_guided = forward_with_noise_level(
                            model, x, guided_noise_level, noise_density=None, randomize_noise=random_noise
                        )
                        un_x[prompt_index] = mask_id
                        unguided_masked_indices_float = (un_x == mask_id).float()
                        unguided_noise_level = unguided_masked_indices_float.mean(
                            dim=1, keepdim=True
                        )
                        logits_unguided = forward_with_noise_level(
                            model, un_x, unguided_noise_level, noise_density=None, randomize_noise=random_noise
                        )
                        logits = logits_unguided + (cfg_scale + 1) * (
                            logits_guided - logits_unguided
                        )
                    else:
                        guided_noise_level=_noise_level_excluding_prompt(x, prompt_index, mask_id)
                        logits_guided = forward_with_noise_level(
                            model, x, guided_noise_level, noise_density=None, randomize_noise=random_noise
                        )
                        logits_unguided = forward_with_noise_level(
                            model, un_x, guided_noise_level, noise_density=None, randomize_noise=random_noise
                        )
                        logits = logits_unguided + (cfg_scale + 1) * (
                            logits_guided - logits_unguided
                        )

                else:
                    if not direct_noise:
                        masked_indices_float = (x == mask_id).float()
                        noise_level = masked_indices_float.mean(dim=1, keepdim=True)
                        logits = forward_with_noise_level(
                            model, x, noise_level, noise_density=None, randomize_noise=random_noise
                        )
                    else:
                        noise_level=_noise_level_excluding_prompt(x, prompt_index, mask_id)
                        logits = forward_with_noise_level(
                            model, x, noise_level, noise_density=None, randomize_noise=random_noise
                        )

            # --- BRANCH 4: DORA_V2 (uses simple set_context_state with noise_level only) ---
            elif finetuning_type == FINETUNING_TYPE.DORA_V2:
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    if not direct_noise:
                        guided_masked_indices_float = (x == mask_id).float()
                        guided_noise_level = guided_masked_indices_float.mean(
                            dim=1, keepdim=True
                        )

                        logits_guided = forward_with_noise_level(
                            model, x, guided_noise_level, noise_density=None, randomize_noise=random_noise
                        )
                        un_x[prompt_index] = mask_id
                        unguided_masked_indices_float = (un_x == mask_id).float()
                        unguided_noise_level = unguided_masked_indices_float.mean(
                            dim=1, keepdim=True
                        )
                        logits_unguided = forward_with_noise_level(
                            model, un_x, unguided_noise_level, noise_density=None, randomize_noise=random_noise
                        )
                        logits = logits_unguided + (cfg_scale + 1) * (
                            logits_guided - logits_unguided
                        )
                    else:
                        guided_noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id)
                        logits_guided = forward_with_noise_level(
                            model, x, guided_noise_level, noise_density=None, randomize_noise=random_noise
                        )
                        logits_unguided = forward_with_noise_level(
                            model, un_x, guided_noise_level, noise_density=None, randomize_noise=random_noise
                        )
                        logits = logits_unguided + (cfg_scale + 1) * (
                            logits_guided - logits_unguided
                        )
                else:
                    if not direct_noise:
                        masked_indices_float = (x == mask_id).float()
                        noise_level = masked_indices_float.mean(dim=1, keepdim=True)
                        logits = forward_with_noise_level(
                            model, x, noise_level, noise_density=None, randomize_noise=random_noise
                        )
                    else:
                        noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id)
                        logits = forward_with_noise_level(
                            model, x, noise_level, noise_density=None, randomize_noise=random_noise
                        )

            # --- BRANCH 5: BASELINE / OTHERS ---
            else:
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = model(x_).logits
                    if finetuning_type in (FINETUNING_TYPE.PTUNING,FINETUNING_TYPE.PROMPT_TUNING,):
                        adapter_name = model.active_adapter
                        current_config = model.peft_config[adapter_name]
                        num_virtual_tokens = current_config.num_virtual_tokens
                        logits = logits[:, num_virtual_tokens:, :]
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x).logits
                    if finetuning_type in (FINETUNING_TYPE.PTUNING,FINETUNING_TYPE.PROMPT_TUNING,):
                        adapter_name = model.active_adapter
                        current_config = model.peft_config[adapter_name]
                        num_virtual_tokens = current_config.num_virtual_tokens
                        logits = logits[:, num_virtual_tokens:, :]

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], k=int(num_transfer_tokens[j, i])
                )
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
                    
            steps_bar.update(1)
            steps_bar.set_postfix({"Blocks": f"{num_block + 1}/{num_blocks}"})
            
        if till_eos:
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length

            # 1. Check if we need to stop based on the PREVIOUS block's EOS
            # If this flag is True, it means we have just finished generating 
            # the "one extra block" requested.
            if stop_after_current_block:
                # Fill everything from the end of this block to the very end with EOS
                x[:, end_idx:] = tokenizer.eos_token_id
                break

            # 2. Check if the CURRENT block has EOS
            # If found, we do NOT break yet. We set the flag so the loop 
            # runs exactly one more time (for the next block).
            
            if (x[:, start_idx:end_idx] == tokenizer.eos_token_id).any():
                stop_after_current_block = True
        if till_current_eos:
            if stop_after_current_block:
                # import pdb; pdb.set_trace()
                x[:, end_idx:] = tokenizer.eos_token_id
                break
    return x

def _get_adapter_name_from_noise(noise_ratio: float, num_bins: int) -> str:
    """Maps a noise ratio in (0, 1] to a LoRA adapter name."""
    if noise_ratio <= 0.0:
        return "noise_bin_1"
    
    # Clamp noise_ratio to (0, 1]
    noise_ratio = min(max(noise_ratio, 1e-9), 1.0) 
    
    bin_index = int(math.ceil(noise_ratio * num_bins))
    
    # Ensure bin_index is within [1, num_bins]
    bin_index = min(max(1, bin_index), num_bins)
    
    return f"noise_bin_{bin_index}"

@torch.no_grad()
def generate_multi_lora(
    model: PeftModel,
    num_lora_bins: int,
    prompt: torch.Tensor,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    is_main_process: bool = True,
    **kwargs # Accept other args like finetuning_type, etc., even if unused
):
    """
    Generate function modified for Multi-LoRA.
    Dynamically switches adapters based on the current noise level.
    
    Args:
        model: The PeftModel containing multiple 'noise_bin_X' adapters.
        num_lora_bins: The total number of bins (e.g., 5).
        prompt: A tensor of shape (1, L).
        ... (other generation args) ...
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    steps_bar = tqdm(
        total=steps,
        initial=0,
        desc="Steps",
        leave=False,
        disable=not is_main_process,
    )
    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1] + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length :,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        steps_bar.reset()
        for i in range(steps):
            mask_index = (x == mask_id)
            
            # ##############################################################
            # ### START MULTI-LORA DYNAMIC ADAPTER SWITCHING ###
            # ##############################################################
            
            # 1. Calculate the current noise level in the answer part
            current_noise_level = _noise_level_excluding_prompt(x, prompt_index, mask_id).item()
            
            # 2. Get the correct adapter name for this noise level
            adapter_name = _get_adapter_name_from_noise(current_noise_level, num_lora_bins)
            
            try:
                # 3. Set the active adapter
                model.set_adapter(adapter_name)
            except Exception as e:
                if is_main_process:
                    print(f"\n[Warning] Failed to set adapter {adapter_name} for noise {current_noise_level:.4f}. Error: {e}")
            
            # 5. Run the standard forward pass (no more TLORA/NORA logic)
            if cfg_scale > 0.0:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                
                # Model forward pass
                logits = model(x_).logits
                
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                # Model forward pass
                logits = model(x).logits

            # ##############################################################
            # ### END MULTI-LORA DYNAMIC ADAPTER SWITCHING ###
            # ##############################################################

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == "low_confidence":
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )  # b, l
            elif remasking == "random":
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(
                    confidence[j], k=int(num_transfer_tokens[j, i])
                )
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
            steps_bar.update(1)
            steps_bar.set_postfix({"Blocks": f"{num_block + 1}/{num_blocks}"})
    return x

def main():
    device = "cuda:3"

    model = (
        AutoModel.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False
    )

    input_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # out = generate(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    # out = generate_v2(model, input_ids, steps=16, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence_v2')
    out = generate(
        model,
        input_ids,
        steps=16,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence_with_block",
    )
    print(
        tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[
            0
        ]
    )


if __name__ == "__main__":
    main()
