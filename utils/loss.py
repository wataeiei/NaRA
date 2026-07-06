import torch
import torch.nn.functional as F
import random
import math

from config import MODEL_TYPE, get_type, MASK_ID_MAPPING, FINETUNING_TYPE
from .util import (
    shift_logits,
    random_forward_process,
    forward_with_noise_level,
)

def _get_nested(config, keys, default=None):
    cur = config
    for key in keys:
        try:
            cur = cur.get(key, default)
        except AttributeError:
            cur = getattr(cur, key, default)
        if cur is default:
            return default
    return cur


def _build_lift_a_supervision_mask(logits, input_ids, masked_indices, question_length, ratios, config):
    """Approximate LIFT-A: select masked tokens for loss by confidence and noise regime."""
    if not _get_nested(config, ["train", "lift", "enabled"], False):
        return masked_indices

    variant = str(_get_nested(config, ["train", "lift", "variant"], "lift_a")).lower()
    if variant not in {"lift_a", "lifta"}:
        raise ValueError(f"Unsupported LIFT variant for this implementation: {variant}")

    H = float(_get_nested(config, ["train", "lift", "H"], 3))
    if H < 2:
        raise ValueError("train.lift.H must be >= 2")

    select_fraction = float(_get_nested(config, ["train", "lift", "select_fraction"], 0.5))
    select_fraction = min(1.0, max(0.0, select_fraction))
    min_selected_tokens = int(_get_nested(config, ["train", "lift", "min_selected_tokens"], 1))

    supervision_mask = torch.zeros_like(masked_indices)
    with torch.no_grad():
        probs = torch.softmax(logits.float(), dim=-1)
        gt_conf = probs.gather(-1, input_ids.long().unsqueeze(-1)).squeeze(-1)

        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            token_idx = torch.nonzero(masked_indices[b], as_tuple=False).flatten()
            num_masked = int(token_idx.numel())
            if num_masked == 0:
                continue

            prompt_len = int(question_length[b].item()) if torch.is_tensor(question_length) else int(question_length)
            answer_len = max(1, input_ids.shape[1] - prompt_len)
            noise_t = float(num_masked) / float(answer_len)

            if (1.0 / H) <= noise_t < (1.0 - 1.0 / H):
                supervision_mask[b, token_idx] = True
                continue

            k = max(min_selected_tokens, math.ceil(num_masked * select_fraction))
            k = min(num_masked, k)
            conf = gt_conf[b, token_idx]

            if noise_t < (1.0 / H):
                # Low-noise inputs have enough context, so train harder low-confidence tokens.
                chosen_local = torch.topk(conf, k=k, largest=False).indices
            else:
                # High-noise inputs have little context, so train easier high-confidence tokens.
                chosen_local = torch.topk(conf, k=k, largest=True).indices

            supervision_mask[b, token_idx[chosen_local]] = True

    return supervision_mask


def compute_loss_by_config(
    input_ids, denoiser, question_length, config, noise_ratio=None, cached_noise_data=None,answer_length=None
):
    """Select different loss functions based on config file"""
    model_type: MODEL_TYPE = get_type(MODEL_TYPE, config.get("model", None))
    if model_type in [
        MODEL_TYPE.LLADA_INSTRUCT,
        MODEL_TYPE.LLADA_BASE,
    ]:
        return compute_original_llada_loss(
            input_ids, denoiser, question_length, config, noise_ratio, 
            cached_noise_data=cached_noise_data, # <--- Pass it down
            answer_length=answer_length,
        )
    else:
        raise ValueError(f"Unsupported training mode: {config.get('model', None)}")
    




def compute_original_llada_loss(input_ids, denoiser, question_length, config, noise_ratio, cached_noise_data=None,answer_length=None):
    mask_id = MASK_ID_MAPPING[MODEL_TYPE(config.get("model", None))]
    generation_alignment_steps = config.train.get("generation_alignment_steps", None)
    batch_size=config.data.get("batch_size", None)

    if generation_alignment_steps is not None:
        # 1. Judge if the batch_size is 1, if not report an error
        # We check strictly against 1 because variable length alignment usually breaks batching 
        # unless padding is handled very specifically, which is not indicated here.
        if batch_size != 1:
            raise ValueError(f"generation_alignment_steps is set, which requires batch_size=1. Found batch_size={batch_size}")

        # 2. Decide how to cut the input_ids
        current_seq_len = input_ids.shape[-1]
        
        # Ensure question_length is a python scalar for the comparison logic
        q_len_val = question_length.item() if isinstance(question_length, torch.Tensor) else question_length
        
        # Ensure answer_length is a python scalar if provided
        ans_len_val = None
        if answer_length is not None:
            ans_len_val = answer_length.item() if isinstance(answer_length, torch.Tensor) else answer_length

        # 3. Loop over steps, keep satisfying question_length + step <= input_ids.shape[-1]
        valid_steps = [
            step for step in generation_alignment_steps 
            if (q_len_val + step) <= current_seq_len and (ans_len_val is None or step > ans_len_val)
        ]

        # 4. Randomly choose one of these steps to cut the input_ids
        # If no such step exists, we skip this block and use original input_ids (Fallback)
        if valid_steps:
            selected_step = random.choice(valid_steps)
            new_total_length = int(q_len_val + selected_step)
            
            # Slice the input_ids to the new calculated length
            input_ids = input_ids[:, :new_total_length]

    # --- CHANGE START: Check for cache ---
    if cached_noise_data is not None:
        # Reuse the previously generated noise
        noisy_batch, masked_indices, ratios = cached_noise_data
    else:
        # Generate new noise (Original logic)
        noisy_batch, masked_indices, ratios, selected_length = random_forward_process(
            input_ids,
            mask_id=mask_id,
            prompt_lengths=question_length,
            per_example_ratio=True,
            fixed_ratio=noise_ratio,
            config=config,
            answer_length=answer_length,
        )

    # answer_length is typically a Tensor [batch_size] (e.g. [512 - 100])
    answer_length = input_ids.shape[-1] - question_length

    finetuning_method_name = config.get("finetuning_method", None)
    if finetuning_method_name:
        finetuning_type: FINETUNING_TYPE = get_type(
            FINETUNING_TYPE, finetuning_method_name
        )
    else:
        finetuning_type = None
    
    # Prepare return variables
    noise_level_for_loss = None # For logging
    
    # --- 1. VARLENLORA LOGIC ---
    # Check for VARLENLORA (Handling Enum or String attribute)
    is_varlen = (finetuning_type == getattr(FINETUNING_TYPE, "VARLENLORA", "VARLENLORA"))
    
    if is_varlen:
        real_model = denoiser.module if hasattr(denoiser, "module") else denoiser
        
        # Set the Target Length Context

        if hasattr(real_model, "set_target_length"):
            # answer_length is passed as the context.
            # If batch_size > 1, this assumes we want the per-sample length or average.
            # Since VarLenLoRA enforces batch_size=1, this tensor is effectively a scalar.
            if selected_length:
                real_model.set_target_length(selected_length)
            else:
                real_model.set_target_length(answer_length)
                
        logits = denoiser(noisy_batch).logits
        

    # --- 2. NOISE-BASED LORA LOGIC (TLORA, NORA, CLORA, NARA) ---
    elif finetuning_type == FINETUNING_TYPE.NARA:
            real_model = denoiser.module if hasattr(denoiser, "module") else denoiser
            peft_cfg = real_model.peft_config['default']
            
            # Variables to pass to forward
            current_nl = None
            current_nd = None
            
            # Calculate Noise Level
            if getattr(peft_cfg, "direct_noise_level", True):
                calculated_nl = ratios
            else:
                masked_indices_float = masked_indices.float()
                calculated_nl = masked_indices_float.mean(dim=1, keepdim=True)
                
            # Logic for CLORA
            if False:
                input_components = getattr(peft_cfg, "input_components", [])
                if "nl" in input_components:
                    current_nl = calculated_nl
                if "nd" in input_components:
                    radius = getattr(peft_cfg, "density_radius", None)
                    current_nd = calculate_global_mask_density(masked_indices, r=radius)
                    
            elif finetuning_type == FINETUNING_TYPE.NARA:
                input_mode = getattr(peft_cfg, "input_mode", "nl")
                if input_mode == "constant":
                    # Constant mode: no noise info needed
                    current_nl = None
                    current_nd = None
                elif input_mode == "nl":
                    current_nl = calculated_nl
                elif input_mode == "nd":
                    radius = peft_cfg.density_radius
                    current_nd = calculate_global_mask_density(masked_indices, r=radius)
                elif input_mode == "both":
                    current_nl = calculated_nl
                    radius = peft_cfg.density_radius
                    current_nd = calculate_global_mask_density(masked_indices, r=radius)
        
            # Logic for NORA/Others
            else:
                input_mode = getattr(peft_cfg, "input_mode", "noise_level")
                if input_mode in ["noise_density", "both"]:
                    radius = peft_cfg.density_radius
                    current_nd = calculate_global_mask_density(masked_indices, r=radius)

                if input_mode == "noise_level":
                    current_nl = calculated_nl
                elif input_mode == "noise_density":
                    current_nd = current_nd
                elif input_mode == "both":
                    current_nl = calculated_nl
                    current_nd = current_nd

            # Forward Pass using helper
            logits = forward_with_noise_level(
                denoiser, 
                noisy_batch, 
                noise_level=current_nl, 
                noise_density=current_nd
            )
            
            noise_level_for_loss = calculated_nl

    # --- 3. DORA_V2 LOGIC (uses set_context_state with only noise_level) ---
    elif finetuning_type == FINETUNING_TYPE.DORA_V2:
        real_model = denoiser.module if hasattr(denoiser, "module") else denoiser

        # Calculate noise level from ratios
        calculated_nl = ratios

        # Set context state for DoRA_V2 (only noise_level)
        if hasattr(real_model, "set_context_state"):
            real_model.set_context_state(noise_level=calculated_nl)

        logits = denoiser(noisy_batch).logits
        noise_level_for_loss = calculated_nl

    # --- 4. BASELINE LOGIC ---
    else:
        logits = denoiser(noisy_batch).logits
        # import pdb; pdb.set_trace()
    if finetuning_type in [getattr(FINETUNING_TYPE, 'PTUNING', None), getattr(FINETUNING_TYPE, 'PROMPT_TUNING', None)]:
        num_virtual_tokens = config.finetuning_parameters.get("num_virtual_tokens",None)
        logits = logits[:, num_virtual_tokens:, :]
    supervision_mask = _build_lift_a_supervision_mask(
        logits=logits,
        input_ids=input_ids,
        masked_indices=masked_indices,
        question_length=question_length,
        ratios=ratios,
        config=config,
    )
    if not torch.any(supervision_mask):
        supervision_mask = masked_indices

    token_loss = F.cross_entropy(logits[supervision_mask], input_ids[supervision_mask].long(), reduction="none") / answer_length
    lift_selected_tokens = int(supervision_mask.sum().detach().item())
    lift_masked_tokens = int(masked_indices.sum().detach().item())

    use_cross_entropy = config.eval.get("use_cross_entropy", False) if hasattr(config, "eval") else False
    # import pdb; pdb.set_trace()
    if use_cross_entropy:
        if not getattr(compute_original_llada_loss, "has_printed_ce_info", False):
            print(f"Info: Using Cross Entropy Loss (skipping division by noise ratio).")
            compute_original_llada_loss.has_printed_ce_info = True
        losses = {
            "loss": token_loss.sum(),
            "masked_indices": supervision_mask,
            "ratios": ratios,
            "lift_selected_tokens": lift_selected_tokens,
            "lift_masked_tokens": lift_masked_tokens,
        }
        return losses
    # Calculate final loss value    
    # --- CHANGE START: Pack the noise data into the return dict ---
    # We return the noise data so the main loop can cache it if needed
    noise_data_payload = (noisy_batch, masked_indices, ratios)
    # Construct Return Dictionary
    if False:
        losses = {
            "loss": token_loss.sum()/ratios,
            "noise_level": noise_level_for_loss, 
            "masked_indices": supervision_mask, 
            "noise_data_cache": noise_data_payload,
            "lift_selected_tokens": lift_selected_tokens,
            "lift_masked_tokens": lift_masked_tokens,
        }
    else:
        # VarLenLoRA falls here (no noise level to log specific to the adapter)
        losses = {
            "loss": token_loss.sum()/ratios,
            "masked_indices": supervision_mask,
            "noise_data_cache": noise_data_payload,
            "lift_selected_tokens": lift_selected_tokens,
            "lift_masked_tokens": lift_masked_tokens,
        }
    return losses


def compute_original_dream_loss(
    input_ids,
    denoiser,
    question_length,
    mask_id,
):
    noisy_batch, masked_indices = random_forward_process(
        input_ids, mask_id=mask_id, prompt_lengths=question_length
    )
    noisy_batch = noisy_batch.to(denoiser.device)
    logits = denoiser(noisy_batch).logits
    logits = shift_logits(logits)
    token_loss = F.cross_entropy(
        logits[masked_indices], input_ids[masked_indices], reduction="none"
    )
    losses = {
        "loss": token_loss.mean(),
    }
    return losses


def show_denoising_process(
    input_ids, question_length, tokenizer, denoiser, mask_id=126336
):
    """
    Debugging function to show the raw data, masked version, and model output.

    Args:
        input_ids: torch.Tensor (B, L) - input token ids
        question_length: int - prefix/question length
        tokenizer: HuggingFace tokenizer
        denoiser: torch.nn.Module - denoising model
        mask_id: int - special id used for masking
    """
    denoiser.eval()  # no dropout etc.

    # -----------------------
    # 1. Decode raw original input
    # -----------------------
    raw_questions = tokenizer.batch_decode(
        input_ids[:, :question_length], skip_special_tokens=True
    )
    raw_answers = tokenizer.batch_decode(
        input_ids[:, question_length:], skip_special_tokens=True
    )

    # -----------------------
    # 2. Mask the inputs
    # -----------------------
    noisy_batch, _ = random_forward_process(
        input_ids,
        mask_id=mask_id,
        prompt_lengths=torch.tensor([question_length] * input_ids.size(0)),
        fixed_ratio=1.0,
    )
    noisy_decoded = tokenizer.batch_decode(noisy_batch, skip_special_tokens=True)

    # -----------------------
    # 3. Run model on noisy input
    # -----------------------
    noisy_batch = noisy_batch.to(denoiser.device)
    logits = denoiser(noisy_batch).logits
    predictions = logits.argmax(dim=-1)
    pred_answers = tokenizer.batch_decode(
        predictions[:, question_length:], skip_special_tokens=True
    )

    # -----------------------
    # 4. Pretty print
    # -----------------------
    for i in range(input_ids.size(0)):
        print("=" * 80)
        print(f"🔹 Sample {i}")
        print("-" * 80)
        print("Raw Data:")
        print(f"  Question   : {raw_questions[i]}")
        print(f"  Answer     : {raw_answers[i]}")
        print("-" * 80)
        print("Masked Input:")
        print(f"  Masked     : {noisy_decoded[i]}")
        print("-" * 80)
        print("Model Prediction:")
        print(f"  Pred Ans   : {pred_answers[i]}")
        print("=" * 80)
        
def calculate_local_mask_density(mask_tensor: torch.BoolTensor, r: int) -> torch.Tensor:
    """
    Calculates the local mask density for each token in a batch of sequences.
    
    Density = (Count of masks within radius r) / (Count of total valid tokens within radius r)
    *Note: The token itself is excluded from the count.*

    Args:
        mask_tensor: A BoolTensor of shape (Batch, Length) or (1, Length).
                     True indicates a mask.
        r: Integer radius (>= 1).

    Returns:
        A FloatTensor of shape (Batch, Length) containing density values between 0.0 and 1.0.
    """
    if r < 1:
        raise ValueError("r must be an integer greater than or equal to 1")

    # 1. Prepare Input
    # Convert Bool (True/False) to Float (1.0/0.0)
    # conv1d expects shape (Batch, Channel, Length). 
    # Input is (B, L), so we unsqueeze to make it (B, 1, L).
    mask_float = mask_tensor.float().unsqueeze(1) 
    
    # 2. Create the Convolution Kernel
    # Size is 2*r + 1 (Left r + Center + Right r)
    kernel_size = 2 * r + 1
    kernel = torch.ones((1, 1, kernel_size), device=mask_tensor.device)
    
    # Set the center to 0 to exclude the current token from the calculation
    kernel[0, 0, r] = 0.0

    # 3. Calculate Numerator (Count of Masks nearby)
    # We use padding=r to maintain the same sequence length.
    neighbor_mask_count = F.conv1d(mask_float, kernel, padding=r)

    # 4. Calculate Denominator (Count of Valid Neighbors nearby)
    # Create a tensor of all 1s to represent "existence of a token"
    ones_tensor = torch.ones_like(mask_float)
    neighbor_total_count = F.conv1d(ones_tensor, kernel, padding=r)

    # 5. Calculate Density
    # We clamp the denominator to a minimum of 1.0 to avoid division by zero 
    # for very short sequences where neighbor count might be 0.
    density = neighbor_mask_count / neighbor_total_count.clamp(min=1.0)

    # Remove the extra channel dimension: (B, 1, L) -> (B, L)
    return density.squeeze(1)

def calculate_global_mask_density(mask_tensor: torch.BoolTensor, r: int) -> torch.Tensor:
    """
    Calculates the global mask density for each sequence in the batch.
    Defined as the average local mask density specifically at the positions where masks exist.

    Args:
        mask_tensor: A BoolTensor of shape (Batch, Length).
        r: Integer radius.

    Returns:
        A FloatTensor of shape (Batch,) containing the average density for each sequence.
        Returns 0.0 for sequences that have no masks.
    """
    # 1. Get local densities for all positions
    # Shape: (B, L)
    local_densities = calculate_local_mask_density(mask_tensor, r)
    # 2. Convert mask to float for arithmetic operations
    # Shape: (B, L)
    mask_float = mask_tensor.float()

    # 3. Sum the densities ONLY at mask positions
    # We multiply by mask_float so non-mask positions become 0.0
    # Summing along dim=1 gives the total density score for the sequence
    # Shape: (B,)
    sum_densities = (local_densities * mask_float).sum(dim=1)

    # 4. Count the number of masks in each sequence
    # Shape: (B,)
    num_masks = mask_float.sum(dim=1)

    # 5. Calculate Average
    # Avoid division by zero: if num_masks is 0, the result should be 0.0.
    # We clamp denominator to 1.0, perform division, then mask out the originally 0-count entries.
    avg_densities = sum_densities / num_masks.clamp(min=1.0)
    
    # Explicitly set density to 0.0 where there were no masks (to handle the clamped 0/1 case correctly if needed, 
    # though sum_densities would be 0 anyway, so 0/1=0. This line is a safety/clarity guard).
    avg_densities = torch.where(num_masks > 0, avg_densities, torch.zeros_like(avg_densities))

    return avg_densities
