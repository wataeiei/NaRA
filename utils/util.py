import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import os
import random

def random_forward_process(
    input_ids: torch.Tensor,
    mask_id: int,
    prompt_lengths: torch.Tensor,
    per_example_ratio: bool = False,
    fixed_ratio: float | None = None,  # for debug
    ensure_at_least_one: bool = True,
    config: object | None = None,
    answer_length: torch.Tensor | None = None,
):
    """
    Applies random masking to input_ids. 
    Can dynamically truncate sequence effective length based on config.train.random_length.
    """
    device = input_ids.device
    B, L = input_ids.shape
    
    # Check if random length logic is enabled via config
    use_random_length = False
    if config is not None and config.train.random_length:
        use_random_length = True

    # Initialize selected_lengths return value
    selected_length = None
    
    # Determine the effective end of the sequence (current_L)
    current_L = L

    if use_random_length and answer_length is not None:
        # 1. Compute maximum required length across the whole batch
        max_min_len = (prompt_lengths + answer_length).max().item()
        
        # Ensure max_min_len doesn't exceed physical size L
        max_min_len = int(min(max_min_len, L))
        
        # 2. Randomly select ONE length between max_min_len and L for the whole batch
        if max_min_len < L:
            current_L = torch.randint(max_min_len, L + 1, (1,), device=device).item()
        else:
            current_L = L
            
        selected_length = current_L
    # Determine Masking Ratios
    if fixed_ratio is not None:
        ratios = torch.full((B,), float(fixed_ratio), device=device)
    else:
        if per_example_ratio:
            ratios = torch.rand(B, device=device)  # One ratio per sample
        else:
            r = torch.rand(1, device=device)  # One ratio shared across batch
            ratios = r.expand(B)

    masked_indices = torch.zeros((B, L), dtype=torch.bool, device=device)

    for b in range(B):
        start = int(prompt_lengths[b].item())
        
        # Calculate available tokens for masking based on the batch-wide current_L
        avail = current_L - start
        
        if avail <= 0:
            continue
            
        k = int(torch.ceil(ratios[b] * avail).item())

        if ensure_at_least_one and avail > 0:
            k = max(1, min(avail, k))
        else:
            k = min(avail, max(0, k))

        if k == 0:
            continue

        # Generate mask indices within the range [start, current_L)
        idx = torch.randperm(avail, device=device)[:k] + start
        masked_indices[b, idx] = True

    noisy_batch = input_ids.clone()
    noisy_batch[masked_indices] = mask_id


    return noisy_batch, masked_indices, ratios.mean(), selected_length


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def shift_logits(logits):
    shifted_logits = torch.zeros_like(logits)
    shifted_logits[:, 1:, :] = logits[:, :-1, :]
    shifted_logits[:, 0, :] = 1.0

    return shifted_logits


def get_accelerator(config):
    """
    Initialize and return a Hugging Face Accelerator along with the output directory path.

    """
    # ---- Validate required global config ----
    try:
        root_path = config.paths.experiment
    except AttributeError:
        raise KeyError("Please specify global_config.paths.experiment")

    # ---- Build paths ----
    output_dir = os.path.join(root_path, config.train.exp_name, config.train.output_dir)
    logging_dir = os.path.join(output_dir, config.train.logging_dir)

    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)

    # ---- Project configuration ----
    project_config = ProjectConfiguration(
        project_dir=output_dir,
        logging_dir=logging_dir,
    )

    # ---- Accelerator ----
    accelerator = Accelerator(
        log_with=None if config.train.report_to == "no" else config.train.report_to,
        mixed_precision=config.train.mixed_precision,
        project_config=project_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )

    return accelerator, output_dir


_random_noise_print_count = 0

def forward_with_noise_level(model, noisy_batch, noise_level, noise_density=None, randomize_noise=False):
    """
    Performs a forward pass with a specified or randomized noise level (and density).
    """
    global _random_noise_print_count

    effective_noise_density = noise_density

    if randomize_noise:
        effective_noise_level = 1.0 - random.random()
    else:
        effective_noise_level = noise_level

    # Get the actual model, handling DataParallel/DDP wrappers
    real_model = model.module if hasattr(model, "module") else model

    # --- CHANGED HERE FOR CLORA/DORA_V2 SUPPORT ---
    # Check for set_context_state (CLoRA, NARA, DoRA_V2)
    if hasattr(real_model, "set_context_state"):
        # Try CLoRA/NARA style (noise_level + noise_density)
        import inspect
        sig = inspect.signature(real_model.set_context_state)
        if 'noise_density' in sig.parameters:
            real_model.set_context_state(noise_level=effective_noise_level, noise_density=effective_noise_density)
        else:
            # DoRA_V2 style (only noise_level)
            real_model.set_context_state(noise_level=effective_noise_level)
    # Fallback for NORA's set_noise_state
    elif hasattr(real_model, "set_noise_state"):
        real_model.set_noise_state(noise_level=effective_noise_level, noise_density=effective_noise_density)
    # Legacy fallback
    elif hasattr(real_model, "set_noise_level"):
        real_model.set_noise_level(noise_level=effective_noise_level)
    # --------------------
    # Perform the forward pass
    if randomize_noise and _random_noise_print_count < 3:
        print(f"[RandomNoise] Using randomized noise level: {effective_noise_level:.4f} (original: {noise_level})")
        _random_noise_print_count += 1
    
    # 🌟 修改这里：确保 noisy_batch 在送入模型前一定是 Long 类型
    if isinstance(noisy_batch, torch.Tensor):
        noisy_batch = noisy_batch.long()
    elif isinstance(noisy_batch, dict) and "input_ids" in noisy_batch:
        # 如果 noisy_batch 是一个字典（HuggingFace 常见格式）
        noisy_batch["input_ids"] = noisy_batch["input_ids"].long()
        

    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16, cache_enabled=True):
        logits = model(noisy_batch).logits
        
    return logits

    # Reset context
    # if hasattr(real_model, "set_context_state"):
    #     real_model.set_context_state(noise_level=None, noise_density=None)
    # elif hasattr(real_model, "set_noise_state"):
    #     real_model.set_noise_state(noise_level=None, noise_density=None)
    # elif hasattr(real_model, "set_noise_level"):
    #     real_model.set_noise_level(noise_level=None)



