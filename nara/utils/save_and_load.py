# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from peft.config import PeftType, PromptLearningConfig

def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Return only PEFT params for NARA:
      - LoRA A/B (lora_*)
      - NARA: global mapper (global_mapper.*) 
      - NARA: embeddings (embedding_layers.*) -> Includes Gaussian buffers (W)
      - optional bias
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()

    # Ensure NARA is recognized
    nara_type = getattr(PeftType, "NARA", "NARA")

    if config.peft_type == nara_type:
        bias = config.bias

        def _is_curr_adapter(k: str) -> bool:
            # Matches .default. or default_ (for embeddings) or ends with .default
            return (f".{adapter_name}." in k) or (f"{adapter_name}_" in k) or k.endswith(f".{adapter_name}")

        to_return = {}

        # --- base selection ---
        for k, v in state_dict.items():
            # 1. Standard LoRA keys (inside layers)
            is_lora = "lora_" in k

            # 2. NARA keys (Global modules at model root)
            is_nara_global = "global_mapper" in k
            is_nara_embed = "embedding_layers" in k
            is_nara_constant = "constant_c" in k

            # Note: adapter_lambdas logic deleted for NARA

            if is_lora or is_nara_global or is_nara_embed or is_nara_constant:
                # Only save if it belongs to the current adapter
                if _is_curr_adapter(k):
                    to_return[k] = v
            
            elif bias == "all" and "bias" in k:
                to_return[k] = v
            elif bias == "lora_only" and "lora_" in k and _is_curr_adapter(k):
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in state_dict:
                    to_return[bias_name] = state_dict[bias_name]

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif isinstance(config, PromptLearningConfig):
        to_return = {}
        if config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
    else:
        raise NotImplementedError

    # include modules_to_save
    if getattr(model, "modules_to_save", None) is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    # --- CLEANING KEYS (Removing Adapter Name) ---
    cleaned = {}
    suffix = f".{adapter_name}"      # e.g., ".default"
    prefix_embed = f"{adapter_name}_" # e.g., "default_" for embeddings
    
    for k, v in to_return.items():
        # 1. Handle standard LoRA and NARA Global Mapper
        #    Pattern: ...module.default.weight -> ...module.weight
        # if suffix in k:
        #     if "lora_" in k or "global_mapper" in k:
        #         cleaned[k.replace(suffix, "")] = v
        #         continue
        # 1. Handle standard LoRA (inside layers)
        if "lora_" in k:
            if suffix in k:
                cleaned[k.replace(suffix, "")] = v
            else:
                cleaned[k] = v
            continue

        # 2. Handle NARA Global Mapper (root module)
        #    Need to convert adapter_name_groupname to groupname
        if "global_mapper" in k:
            parts = k.split(".")
            if "global_mapper" in parts:
                idx = parts.index("global_mapper")
                # Get the next segment, e.g., "default_qk" or "default"
                mapper_key = parts[idx + 1]
                # import pdb; pdb.set_trace()
                if mapper_key == adapter_name:
                    # Case A: Legacy mode (Key is "default") -> remove it
                    parts.pop(idx + 1)
                    cleaned[".".join(parts)] = v
                elif mapper_key.startswith(f"{adapter_name}_"):
                    # Case B: Group mode (Key is "default_qk") -> convert to "qk"
                    group_name = mapper_key[len(adapter_name) + 1:] # Remove prefix
                    parts[idx + 1] = group_name
                    cleaned_key = ".".join(parts)
                    cleaned[cleaned_key] = v
                    # print(f"[NARA Save] Mapping Group Key: '{k}' -> '{cleaned_key}'")
                else:
                    cleaned[k] = v
            continue
        # 2. Handle NARA Embeddings (Buffers like W)
        #    Pattern: embedding_layers.default_NL_emb.W -> embedding_layers.NL_emb.W
        if "embedding_layers" in k and prefix_embed in k:
            # Only strip the adapter name specifically after 'embedding_layers.'
            parts = k.split(".")
            if "embedding_layers" in parts:
                idx = parts.index("embedding_layers")
                # The next part is 'default_NL_emb' -> convert to 'NL_emb'
                if len(parts) > idx + 1 and parts[idx + 1].startswith(prefix_embed):
                    parts[idx + 1] = parts[idx + 1].replace(prefix_embed, "", 1)
            cleaned[".".join(parts)] = v
            continue

        # 3. Handle NARA constant_c parameter
        #    Pattern: constant_c.default -> constant_c
        if "constant_c" in k:
            if suffix in k:
                cleaned[k.replace(suffix, "")] = v
            else:
                cleaned[k] = v
            continue

        # Default fallback
        cleaned[k] = v

    # Print confirmation for constant mode
    constant_c_keys = [k for k in cleaned.keys() if "constant_c" in k]
    if constant_c_keys:
        print(f"[NARA Save] Constant mode detected. Saving {len(constant_c_keys)} constant_c key(s):")
        for k in constant_c_keys:
            print(f"  - {k}")

    return cleaned


def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    # Check types
    nara_type = getattr(PeftType, "NARA", "NARA")

    if config.peft_type == nara_type:
        peft_model_state_dict = {}
        
        # Get the list of defined group names to distinguish between "Group names" and "model attribute names"
        defined_groups = getattr(config, "mapper_groups", None)
        known_group_names = list(defined_groups.keys()) if defined_groups else []
        
        for k, v in state_dict.items():
            # 1. LORA (Shared logic)
            if "lora_" in k and adapter_name not in k:
                suffix = k.split("lora_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v

            # 2. NARA (Global Mapper)
            # elif "global_mapper" in k and f".{adapter_name}" not in k:
            #     parts = k.split(".")
            #     if "global_mapper" in parts:
            #         idx = parts.index("global_mapper")
            #         parts.insert(idx + 1, adapter_name)
            #         k = ".".join(parts)
            #     peft_model_state_dict[k] = v
            # 2. NARA (Global Mapper)
            elif "global_mapper" in k and f".{adapter_name}" not in k:
                parts = k.split(".")
                if "global_mapper" in parts:
                    idx = parts.index("global_mapper")
                    # Check if the next segment is a known Group Name
                    next_part = parts[idx + 1]

                    if next_part in known_group_names:
                        # Case A: This is a group name (e.g., "qk") -> restore to "default_qk"
                        parts[idx + 1] = f"{adapter_name}_{next_part}"
                        
                        target_key = ".".join(parts)
                        # Debug probe: print conversion process
                        # print(f"[NARA Load] Restoring Group Key: '{k}' -> '{target_key}'")
                        k = target_key
                    else:
                        # Case B: Legacy weight or default group (e.g., "model") -> insert "default"
                        parts.insert(idx + 1, adapter_name)
                    
                    k = ".".join(parts)
                peft_model_state_dict[k] = v    
            # 3. NARA (Embeddings)
            elif "embedding_layers" in k:
                parts = k.split(".")
                if "embedding_layers" in parts:
                    idx = parts.index("embedding_layers")
                    if len(parts) > idx + 1:
                        module_key = parts[idx + 1]
                        if not module_key.startswith(f"{adapter_name}_"):
                            parts[idx + 1] = f"{adapter_name}_{module_key}"
                    k = ".".join(parts)
                peft_model_state_dict[k] = v

            # 4. NARA (constant_c parameter)
            elif "constant_c" in k:
                if adapter_name not in k:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v

            # Note: adapter_lambdas logic deleted for NARA

            else:
                peft_model_state_dict[k] = v

    elif isinstance(config, PromptLearningConfig) or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError
    # import pdb; pdb.set_trace()
    load_result = model.load_state_dict(peft_model_state_dict, strict=False)

    # Diagnostic Logic:
    if len(load_result.unexpected_keys) > 0:
        print(f"\n[NARA Load Check] WARNING: {len(load_result.unexpected_keys)} keys in the checkpoint were unused (Unexpected):")
        for k in load_result.unexpected_keys[:10]:
            print(f"  - {k}")
        if len(load_result.unexpected_keys) > 10: print("  ... (and more)")

    peft_keywords = ["lora_", "global_mapper", "embedding_layers", "constant_c"]
    missing_peft_keys = [k for k in load_result.missing_keys if any(x in k for x in peft_keywords)]

    if len(missing_peft_keys) > 0:
        print(f"\n[NARA Load Check] WARNING: {len(missing_peft_keys)} PEFT parameters in the model were NOT found in the checkpoint (Missing):")
        for k in missing_peft_keys[:10]:
            print(f"  - {k}")
        if len(missing_peft_keys) > 10: print("  ... (and more)")

    # Print confirmation for constant mode
    constant_c_keys = [k for k in peft_model_state_dict.keys() if "constant_c" in k]
    if constant_c_keys:
        print(f"[NARA Load] Constant mode detected. Loading {len(constant_c_keys)} constant_c key(s):")
        for k in constant_c_keys:
            print(f"  - {k}")

    if len(load_result.unexpected_keys) == 0 and len(missing_peft_keys) == 0:
        print("[NARA Load Check] Success: All Adapter keys matched perfectly.")
    # ----------------------------------------------

    if isinstance(config, PromptLearningConfig):
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )