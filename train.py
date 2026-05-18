from utils.util import flatten_dict
from data.main_functions import get_dataloader
from model.get_model import get_model_by_config
from utils.loss import compute_loss_by_config
from eval.eval import evaluate_model
from utils.util import get_accelerator
import math
from transformers import get_linear_schedule_with_warmup

import os
import torch
import argparse
import torch.distributed as dist
from omegaconf import OmegaConf, ListConfig
from tqdm import tqdm
from config import set_seed
import shutil  # ==== BEST-CKPT: for removing previous best
from typing import Dict, List, Optional

DEBUG = False  # no wandb output and ckpt saving
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["WANDB_MODE"] = "offline"  # Removed: use online mode for LR ablation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_DISABLE_SYSTEM_METRICS"] = "true"

def main(args):
    # Load + merge
    config = OmegaConf.load(args.config)

    # Override lr from command line if provided
    if args.lr is not None:
        config.train.lr = args.lr
        print(f"[CONFIG] Overriding lr from command line: {args.lr}")

    accelerator, output_dir = get_accelerator(config)

    os.environ["WANDB_DIR"] = os.path.join(
        config.paths.experiment, config.train.exp_name
    )
    # ---- Model & tokenizer ----

    denoiser, tokenizer = get_model_by_config(config)

    # ---- Enable debug mode for DoRA_V2 if --debug flag is set ----
    if args.debug:
        real_model = denoiser.module if hasattr(denoiser, "module") else denoiser
        if hasattr(real_model, "set_debug_mode"):
            real_model.set_debug_mode(True)

    # ---- Parameter Statistics for Experiment Logging ----
    def count_peft_params(model):
        """Count PEFT parameter counts for experiment logging."""
        a_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_A' in n and p.requires_grad)
        b_params = sum(p.numel() for n, p in model.named_parameters() if 'lora_B' in n and p.requires_grad)
        hyper_params = sum(p.numel() for n, p in model.named_parameters()
                          if ('global_mapper' in n or 'embedding_layers' in n) and p.requires_grad)
        total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_all = sum(p.numel() for p in model.parameters())
        return {
            "A_params_M": a_params / 1e6,
            "B_params_M": b_params / 1e6,
            "hyper_params_M": hyper_params / 1e6,
            "total_trainable_M": total_trainable / 1e6,
            "total_all_M": total_all / 1e6,
            "trainable_ratio": 100 * total_trainable / total_all if total_all > 0 else 0,
        }

    param_stats = count_peft_params(denoiser)
    if accelerator.is_main_process:
        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
        else:
            gpu_name = "N/A"
            gpu_count = 0
        print(f"[GPU INFO] {gpu_name} x {gpu_count}")
        print(f"[PARAM STATS] A: {param_stats['A_params_M']:.2f}M, B: {param_stats['B_params_M']:.2f}M, Trainable: {param_stats['total_trainable_M']:.2f}M / {param_stats['total_all_M']:.2f}M ({param_stats['trainable_ratio']:.4f}%)")

    # ---- DataLoader(s) ----
    dataloaders = {}
    train_dl, val_dl = get_dataloader(accelerator, tokenizer, config)
    dataloaders["train"], dataloaders["val"] = train_dl, val_dl

    # ---- Training state ----
    state = {
        "global_step": config.train.get("global_step") or 0,
        "global_sample_number": config.train.get("global_sample_number") or 0,
        "global_token_number": config.train.get("global_token_number") or 0,
        "global_update_number": config.train.get("global_update_number") or 0,
        "global_epoch": config.train.get("global_epoch") or 0,
    }

    # ========= Helper: evaluate val loss for a list of fixed noise ratios =========
    def _eval_val_loss_over_noise_levels(
        noise_levels: Optional[List[float]],
    ) -> Dict[str, float]:
        """
        Evaluate the model on the validation set for each user-specified noise ratio.
        - noise_levels: list of floats in (0, 1]; if None or empty, returns {}.

        Returns:
            Dict[str, float]: e.g., {"val_loss/noise_0.25": 0.1234, ...}
        """
        results: Dict[str, float] = {}
        if not noise_levels:
            return results

        # sanitize & unique (keep order)
        cleaned: List[float] = []
        seen = set()
        for x in list(noise_levels):
            # Support OmegaConf ListConfig
            v = float(x)
            # Only evaluate valid noise in (0, 1]
            if (v <= 0.0) or (v > 1.0):
                if accelerator.is_main_process:
                    print(f"[WARN] Skip invalid noise_ratio={v}, must be in (0, 1].")
                continue
            if v not in seen:
                cleaned.append(v)
                seen.add(v)

        if not cleaned:
            return results

        # For each fixed noise ratio, iterate the whole validation set
        # NOTE: This can be expensive if val set is large, but it's the most faithful evaluation.
        for noise_ratio in cleaned:
            running = 0.0
            # A local progress bar per noise for user feedback on main process
            per_noise_bar = tqdm(
                total=len(dataloaders["val"]),
                initial=0,
                desc=f"Val (noise={noise_ratio:.3f})",
                leave=False,
                disable=not accelerator.is_local_main_process,
            )
            for val_batch_num, val_batch in enumerate(dataloaders["val"]):
                with torch.no_grad():
                    input_ids: torch.Tensor = val_batch["data"]
                    question_length = val_batch["question_length"]
                    answer_length = val_batch.get("answer_length", None)
                    # ---> key change: pass a fixed noise_ratio here
                    losses_eval = compute_loss_by_config(
                        input_ids,
                        denoiser,
                        question_length,
                        config=config,
                        noise_ratio=noise_ratio,
                        answer_length=answer_length,
                    )
                    val_loss = losses_eval["loss"]
                    # average across devices for this batch
                    val_loss = accelerator.gather(val_loss.detach()).mean().item()
                    running += val_loss

                per_noise_bar.update(1)
                per_noise_bar.set_postfix(
                    {
                        "loss": round(running / (val_batch_num + 1), 4),
                    }
                )
            per_noise_bar.close()

            avg_loss = running / max(1, len(dataloaders["val"]))
            key = f"val_loss/noise_{noise_ratio:.3f}".rstrip("0").rstrip(".")
            results[key] = avg_loss

        return results

    def _save_model_to(save_path: str):
        if not accelerator.is_main_process or DEBUG:
            return
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        unwrapped = accelerator.unwrap_model(denoiser)
        # CPU state_dict to avoid OOM
        with torch.no_grad():
            cpu_state_dict = {k: v.detach().to("cpu") for k, v in unwrapped.state_dict().items()}

        unwrapped.save_pretrained(
            save_path,
            state_dict=cpu_state_dict,
            safe_serialization=True,
            max_shard_size="5GB",
        )
        try:
            tokenizer.save_pretrained(save_path)
        except Exception:
            pass
        print(f"[CKPT] Saved to: {save_path}")

    # ==== BEST-CKPT: tracking best metric and ckpt path
    metric_name = config.train.eval.metric  # "accuracy" or "loss"
    assert metric_name in ["accuracy", "loss"], "config.train.eval.metric must be 'accuracy' or 'loss'"
    higher_is_better = (metric_name == "accuracy")
    best_metric = None
    best_ckpt_path = None
    best_update_number = None

    def _is_better(curr, best):
        if best is None:
            return True
        return (curr > best) if higher_is_better else (curr < best)

    def _save_best_ckpt(curr_metric):
        nonlocal best_metric, best_ckpt_path, best_update_number
        if not accelerator.is_main_process or DEBUG:
            return

        # keep only one best
        if best_ckpt_path is not None and os.path.isdir(best_ckpt_path):
            try:
                shutil.rmtree(best_ckpt_path)
            except Exception as e:
                print(f"[WARN] Failed to remove previous best ckpt: {best_ckpt_path}. Error: {e}")

        tag_metric = f"{curr_metric:.6f}"
        save_dir_name = f"BEST_{metric_name}_{tag_metric}_seed_{args.seed}_update_{state['global_update_number']}_epoch_{epoch_num}"
        save_path = os.path.join(output_dir, save_dir_name)
        _save_model_to(save_path)

        best_metric = curr_metric
        best_ckpt_path = save_path
        best_update_number = state["global_update_number"]
        print(f"[BEST-CKPT] Saved: {save_dir_name}")

    def _save_final_ckpt():
        if not accelerator.is_main_process or DEBUG:
            return
        final_dir_name = f"FINAL_seed_{args.seed}_epoch_{epoch_num}_update_{state['global_update_number']}"
        final_path = os.path.join(output_dir, final_dir_name)
        _save_model_to(final_path)

        try:
            latest_link = os.path.join(output_dir, "latest_final")
            if os.path.islink(latest_link) or os.path.exists(latest_link):
                os.remove(latest_link)
            os.symlink(final_dir_name, latest_link)
        except Exception:
            with open(os.path.join(output_dir, "LATEST_FINAL.txt"), "w") as f:
                f.write(final_dir_name + "\n")
        print(f"[FINAL-CKPT] Saved: {final_dir_name}")

        # Auto-update ckpt_mapping if specified
        if args.ckpt_mapping:
            _update_ckpt_mapping(final_path)

    def _update_ckpt_mapping(final_ckpt_path: str):
        """Update ckpt_mapping.py with the FINAL checkpoint path (with file lock for concurrency)"""
        if not accelerator.is_main_process or DEBUG:
            return

        mapping_file = args.ckpt_mapping
        if not os.path.exists(mapping_file):
            print(f"[CKPT-MAPPING] Warning: {mapping_file} not found, skipping update")
            return

        import fcntl
        lock_file = mapping_file + ".lock"

        try:
            # Format lr string to match mapping keys
            lr_value = config.train.lr
            if isinstance(lr_value, float):
                if lr_value == 5e-5:
                    lr_str = "5e-5"
                elif lr_value == 1e-4:
                    lr_str = "1e-4"
                elif lr_value == 2e-4:
                    lr_str = "2e-4"
                else:
                    lr_str = f"{lr_value:.0e}".replace("e-0", "e-")
            else:
                lr_str = str(lr_value)

            seed = args.seed
            finetuning_method = config.finetuning_method.upper()  # e.g., "LORA", "NARA"
            task_name = config.task_name.upper()  # e.g., "MATH14K", "CODE_FEEDBACK"

            # Use file lock to prevent concurrent writes
            with open(lock_file, "w") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)  # Acquire exclusive lock
                print(f"[CKPT-MAPPING] Acquired lock for {finetuning_method} {task_name} ({lr_str}, {seed})")
                try:
                    # Read the mapping file
                    with open(mapping_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Find the section for this finetuning type and update only within that section
                    import re

                    # Find the block for this task_type AND finetuning_type combination
                    # Pattern matches: MODEL_TYPE.XXX, TASK_TYPE.YYY, FINETUNING_TYPE.ZZZ, ): { ... }
                    # Note: MODEL_TYPE line is optional (some older mappings might not have it)
                    block_pattern = rf'((?:MODEL_TYPE\.\w+,\s*\n\s*)?TASK_TYPE\.{task_name},\s*\n\s*FINETUNING_TYPE\.{finetuning_method},\s*\n\s*\): \{{\s*\n)(.*?)(\n\s*\}})'

                    def replace_in_block(match):
                        block_start = match.group(1)
                        block_content = match.group(2)
                        block_end = match.group(3)

                        # Replace the specific (lr, seed) entry within this block
                        entry_pattern = rf'(\("{re.escape(lr_str)}", {seed}\): )"[^"]*"'
                        new_block_content = re.sub(entry_pattern, rf'\1"{final_ckpt_path}"', block_content)

                        return block_start + new_block_content + block_end

                    new_content, count = re.subn(block_pattern, replace_in_block, content, flags=re.DOTALL)

                    if count > 0:
                        with open(mapping_file, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"[CKPT-MAPPING] Updated: {task_name} {finetuning_method} ({lr_str}, {seed}) -> {final_ckpt_path}")
                    else:
                        print(f"[CKPT-MAPPING] Warning: No matching block found for {task_name} {finetuning_method}")
                finally:
                    fcntl.flock(lf.fileno(), fcntl.LOCK_UN)  # Release lock
                    print(f"[CKPT-MAPPING] Released lock")

        except Exception as e:
            print(f"[CKPT-MAPPING] Error updating mapping: {e}")

    # ---- Optimizer ----
    if config.finetuning_method == "clora":
        # 1. Resolve c_lr: try to get it, default to standard lr if missing or None
        c_lr = config.train.get("c_lr")
        if c_lr is None:
            c_lr = config.train.lr
            if accelerator.is_main_process:
                print(f"[CLoRA Setup] 'c_lr' is None. Using standard lr: {c_lr}")
        
        c_mapper_params = []
        other_params = []

        if accelerator.is_main_process:
            print(f"[CLoRA Setup] Splitting parameters...")
            print(f" -> Standard LR (AB/Base): {config.train.lr}")
            print(f" -> C-Mapper LR: {c_lr}")

        for name, param in denoiser.named_parameters():
            
            if not param.requires_grad:
                continue
            
            # [MODIFIED LINE] Match 'global_mapper' instead of 'c_mapper'
            if "global_mapper" in name or "embedding_layers" in name:
                c_mapper_params.append(param)
                if accelerator.is_main_process:
                     print(f"    [Group C - High LR] {name}")
            else:
                other_params.append(param)
        
        # 3. Create Parameter Groups
        params_to_learn = [
            {"params": other_params, "lr": config.train.lr},  # Group 0
            {"params": c_mapper_params, "lr": c_lr},          # Group 1
        ]
    else:
        # Standard behavior for other methods
        # We must wrap this in a dict too, so we can remove 'lr' from AdamW below consistently
        params_to_learn = [
            {"params": [p for p in denoiser.parameters() if p.requires_grad], "lr": config.train.lr}
        ]

    # 4. Initialize Optimizer
    # REMOVED global 'lr=' argument. It is now handled inside params_to_learn groups.
    optimizer = torch.optim.AdamW(
        params_to_learn,
        betas=(0.9, 0.95),
        weight_decay=5e-2,
        eps=1e-8,
    )
        
    # params_to_learn = [p for p in denoiser.parameters() if p.requires_grad]
    
    # optimizer = torch.optim.AdamW(
    #     params_to_learn,
    #     lr=config.train.lr,
    #     betas=(0.9, 0.95),
    #     weight_decay=5e-2,
    #     eps=1e-8,
    # )
    # ---- LR Scheduler (with warmup) ----
    grad_acc_steps = getattr(accelerator, "gradient_accumulation_steps", 1)
    update_steps_per_epoch = math.ceil(len(train_dl) / grad_acc_steps)
    total_update_steps = config.train.epoch_num * update_steps_per_epoch

    warmup_steps = int(config.train.get("warmup_steps", 0))
    if warmup_steps == 0:
        warmup_ratio = float(config.train.get("warmup_ratio", 0.1))  # default 10%
        warmup_steps = max(1, int(total_update_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_update_steps
    )
    
    stage_1_limit_step = 0
    if config.finetuning_method in ("clora","nara"):
        stage_1_ratio = float(config.train.get("stage_1", 0.0))
        if 0.0 < stage_1_ratio <= 1.0:
            stage_1_limit_step = int(total_update_steps * stage_1_ratio)
            
            # If we are resuming, check where we are; otherwise start at Stage 1
            current_step = state["global_update_number"]
            if current_step < stage_1_limit_step:
                if accelerator.is_main_process:
                    print(f"[{config.finetuning_method}] Starting in Stage 1 (AB Only) until step {stage_1_limit_step}")
                # We use the raw model before prepare, or unwrap if needed. 
                # Since this is before prepare, 'denoiser' is the raw model.
                if hasattr(denoiser, "set_training_stage"):
                    denoiser.set_training_stage(1)
            else:
                 if accelerator.is_main_process:
                    print(f"[{config.finetuning_method}] Resuming/Starting in Stage 2 (ACB + Lambda)")
                 if hasattr(denoiser, "set_training_stage"):
                    denoiser.set_training_stage(2)
    # ---- Accelerator preparation ----
    # for name, param in denoiser.named_parameters():
    #     if "lora_" in name:  # LoRA weights typically include "lora_A" / "lora_B" / "lora_C", etc.
    #         print(f"Before accelerate prepare {name}: {param.dtype}, device={param.device}")
    if "val" in dataloaders:
        denoiser, dataloaders["train"], dataloaders["val"], optimizer, scheduler = (
            accelerator.prepare(
                denoiser, dataloaders["train"], dataloaders["val"], optimizer, scheduler
            )
        )
    else:
        denoiser, dataloaders["train"], optimizer, scheduler = accelerator.prepare(
            denoiser, dataloaders["train"], optimizer, scheduler
        )
    # for name, param in denoiser.named_parameters():
    #     if "lora_" in name:  # LoRA weights typically include "lora_A" / "lora_B" / "lora_C", etc.
    #         print(f"After accelerate prepare {name}: {param.dtype}, device={param.device}")
    # import pdb; pdb.set_trace()
    # ---- Logging / tracking ----
    resume_updates = int(state.get("global_update_number", 0))
    if resume_updates > 0:
        for _ in range(resume_updates):
            scheduler.step()
    run_name = str(config.train.exp_name)+f"_seed_{args.seed}"
    if config.finetuning_method in ("lora","dora","pissa"):
        tags=[
            f"task={config.task_name}",
            f"r={config.finetuning_parameters.r}",
            f"epoches={config.train.epoch_num}",
            f"lr={config.train.lr}",
            f"seed={args.seed}"
        ]
    elif config.finetuning_method in ("ptuning"):
        tags=[
            f"task={config.task_name}",
            f"num_virtual_tokens={config.finetuning_parameters.num_virtual_tokens}",
            f"encoder_hidden_size={config.finetuning_parameters.encoder_hidden_size}",
            f"encoder_reparameterization_type={config.finetuning_parameters.encoder_reparameterization_type}",
            f"epoches={config.train.epoch_num}",
            f"lr={config.train.lr}",
            f"seed={args.seed}"
        ]
    elif config.finetuning_method in ("prefix_tuning",):
        tags=[
            f"task={config.task_name}",
            f"num_virtual_tokens={config.finetuning_parameters.num_virtual_tokens}",
            f"encoder_hidden_size={config.finetuning_parameters.encoder_hidden_size}",
            f"epoches={config.train.epoch_num}",
            f"lr={config.train.lr}",
            f"seed={args.seed}"
        ]
    elif config.finetuning_method in ("prompt_tuning",):
        tags=[
            f"task={config.task_name}",
            f"num_virtual_tokens={config.finetuning_parameters.num_virtual_tokens}",
            f"prompt_tuning_init={config.finetuning_parameters.prompt_tuning_init}",
            f"epoches={config.train.epoch_num}",
            f"lr={config.train.lr}",
            f"seed={args.seed}"
        ]
    elif config.finetuning_method in ("dora_local",):
        tags=[
            f"task={config.task_name}",
            f"r={config.finetuning_parameters.r}",
            f"epoches={config.train.epoch_num}",
            f"lr={config.train.lr}",
            f"seed={args.seed}"
        ]
    elif config.finetuning_method in ("dora_v2",):
        tags=[
            f"task={config.task_name}",
            f"r={config.finetuning_parameters.r}",
            f"epoches={config.train.epoch_num}",
            f"lr={config.train.lr}",
            f"seed={args.seed}"
        ]
    else:
        tags = [
            f"task={config.task_name}",
            f"H={config.finetuning_parameters.get('fnn_hidden_size', 'NA')}",
            f"r={config.finetuning_parameters.r_ab}",
            f"epoches={config.train.epoch_num}",
            f"lr={config.train.lr}",
            f"seed={args.seed}"
        ]
    if not DEBUG:
        accelerator.init_trackers(
            project_name=str(config.train.wandb_proj), 
            config=flatten_dict(config),
            init_kwargs={
                "wandb": {
                    "name": run_name[:128],
                    "tags": tags,      
                }
            },
        )
    if accelerator.is_main_process:
        # [MODIFIED START] Handle list-of-dicts structure
        trainable_params = 0
        for group in params_to_learn:
            # group is a dict: {'params': [tensor1, tensor2...], 'lr': ...}
            trainable_params += sum(p.numel() for p in group["params"])
        # [MODIFIED END]

        # GPU info for logging
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "N/A"

        accelerator.log(
            {
                "trainable_params": int(trainable_params),
                "trainable_params_M": trainable_params / 1e6,
                "trainable_params_B": trainable_params / 1e9,
                "total_params_M": param_stats["total_all_M"],
                "trainable_ratio_percent": param_stats["trainable_ratio"],
                "gpu_name": gpu_name,
            },
            step=state["global_step"],
        )

    # ---- Progress bar ----
    progress_bar = tqdm(
        total=len(dataloaders["train"]),
        initial=state["global_step"] % len(dataloaders["train"]),
        desc="Samples",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )
    logged_loss = None
    logged_noise_level = None
    # removed: saved_global_update_number  # ==== BEST-CKPT: no periodic saving anymore
    evaled_global_update_number = []
    accum_loss = 0.0
    accum_count = 0

    # ---- Read fixed noise levels from config (list in (0,1]) ----
    # Expected path: config.train.eval.noise_levels (e.g., [0.25, 0.5, 0.75, 0.8, 0.9, 1.0])
    fixed_noise_levels = None
    try:
        maybe_levels = config.train.eval.get("noise_levels", None)
        if isinstance(maybe_levels, ListConfig):
            fixed_noise_levels = list(maybe_levels)
        elif isinstance(maybe_levels, (list, tuple)):
            fixed_noise_levels = list(maybe_levels)
        else:
            fixed_noise_levels = None
    except Exception:
        fixed_noise_levels = None

    # ---- Experiment Logging: Start timing and reset VRAM stats ----
    import time
    from datetime import datetime

    training_start_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    # Store initial config for CSV logging
    exp_config = {
        "exp_id": f"{config.finetuning_method}_{config.task_name}_lr{config.train.lr}_seed_{args.seed}",
        "method": config.finetuning_method,
        "task": config.task_name,
        "model": config.model,
        "seed": args.seed,
        "lr": config.train.lr,
        "rank": config.finetuning_parameters.get("r", config.finetuning_parameters.get("r_ab", "NA")),
        "epochs": config.train.epoch_num,
        "batch_size": config.data.batch_size,
        "gradient_accumulation_steps": config.train.gradient_accumulation_steps,
    }

    for epoch_num in range(state["global_epoch"] + 1, config.train.epoch_num + 1):

        progress_bar.reset()
        for batch_num, batch in enumerate(dataloaders["train"]):

            # Debug mode: stop after 5 batches
            if args.debug and batch_num >= 5:
                if accelerator.is_main_process:
                    print(f"[DEBUG] Stopping after {batch_num} batches (debug mode)")
                break

            logs = {}

            accelerator.wait_for_everyone()
            # ---- Evaluation ----
            if (state["global_update_number"] % config.train.eval_every == 0) and (
                state["global_update_number"] not in evaled_global_update_number
            ):
                if not (
                    state["global_update_number"] == 0
                    and not config.train.eval_from_start
                ):
                    denoiser.eval()
                    evaled_global_update_number.append(state["global_update_number"])

                    if metric_name == "accuracy":
                        val_metrics = evaluate_model(
                            accelerator, dataloaders["val"], denoiser, tokenizer, config
                        )
                        total_correct = (
                            accelerator.gather(val_metrics["num_correct"]).sum().item()
                        )
                        total_samples = (
                            accelerator.gather(val_metrics["num_samples"]).sum().item()
                        )
                        accuracy = (
                            total_correct / total_samples if total_samples > 0 else 0.0
                        )
                        logs["val_accuracy"] = accuracy

                        # ==== BEST-CKPT: save if better
                        if _is_better(accuracy, best_metric):
                            _save_best_ckpt(accuracy)

                    elif metric_name == "loss":
                            # --------- Part A: "random noise" validation ---------
                            logs["val_loss"] = 0.0
                            # --- CHANGE START: Initialize Cache ---
                            # Map: batch_index -> (noisy_batch, masked_indices, ratios)
                            val_noise_cache = {} 
                            use_fixed_batch = config.train.eval.get("use_fixed_batch", False) 
                            # --- CHANGE END ---
                            eval_progress_bar = tqdm(
                                total=len(dataloaders["val"]),
                                initial=0,
                                desc="Val Samples",
                                leave=False,
                                disable=not accelerator.is_local_main_process,
                            )
                            for val_epoch in range(config.train.eval.eval_epoches_num):  # can extend to multiple epochs if needed
                                eval_progress_bar.reset()
                                for val_batch_num, val_batch in enumerate(dataloaders["val"]):
                                    with torch.no_grad():
                                        input_ids: torch.Tensor = val_batch["data"]
                                        question_length = val_batch["question_length"]
                                        answer_length = val_batch.get("answer_length", None)
                                        # --- CHANGE START: Retrieve from cache if enabled ---
                                        cached_data = None
                                        if use_fixed_batch and (val_batch_num in val_noise_cache):
                                            cached_data = val_noise_cache[val_batch_num]
                                        # --- CHANGE END ---
                                        losses_eval = compute_loss_by_config(
                                            input_ids,
                                            denoiser,
                                            question_length,
                                            config=config,
                                            cached_noise_data=cached_data, # <--- Pass to function
                                            answer_length=answer_length,
                                        )
                                        # --- CHANGE START: Save to cache if enabled and empty ---
                                        if use_fixed_batch and (val_batch_num not in val_noise_cache):
                                            # We extract the noise data returned by the function
                                            # Ensure you use .detach() or clone to prevent graph retention if needed, 
                                            # though typically for val inference it is okay.
                                            val_noise_cache[val_batch_num] = losses_eval["noise_data_cache"]
                                        # --- CHANGE END ---
                                        val_loss = losses_eval["loss"]
                                        val_loss = (
                                            accelerator.gather(val_loss.detach()).mean().item()
                                        )
                                        logs["val_loss"] += val_loss
                                    eval_progress_bar.update(1)
                                    eval_progress_bar.set_postfix(
                                        {
                                            "loss": round(
                                                logs["val_loss"] / (val_epoch*len(dataloaders["val"])+(val_batch_num + 1)), 4
                                            ),
                                            "epoch": val_epoch,
                                        }
                                    )
                            logs["val_loss"] = logs["val_loss"] / (len(dataloaders["val"])*(config.train.eval.eval_epoches_num))
                            eval_progress_bar.close()

                            # ==== BEST-CKPT: save if better (lower loss)
                            if _is_better(logs["val_loss"], best_metric):
                                _save_best_ckpt(logs["val_loss"])

                            # --------- Part B evaluate fixed noise levels and log to wandb ---------
                            # Only run if the user provided a list in config.train.eval.noise_levels
                            per_noise_logs = {}
                            if fixed_noise_levels:
                                per_noise_logs = _eval_val_loss_over_noise_levels(
                                    fixed_noise_levels
                                )
                                # Merge per-noise losses into logs (e.g., "val_loss/noise_0.25": 0.123)
                                logs.update(per_noise_logs)

            if accelerator.is_main_process:
                logs["loss"] = logged_loss if logged_loss else None
                if config.finetuning_method in ["tlora"]:
                    logs["noise_level"] = (
                        logged_noise_level if logged_noise_level else None
                    )
                logs["epoch"] = epoch_num
                logs["global_step"] = state["global_step"]
                logs["global_sample_number"] = state["global_sample_number"]
                logs["global_token_number"] = state["global_token_number"]
                logs["global_update_number"] = state["global_update_number"]
                logs["lr"] = optimizer.param_groups[0]["lr"]
                if len(optimizer.param_groups) > 1:
                    logs["lr_c_mapper"] = optimizer.param_groups[1]["lr"]

                # optional: log current best
                if best_metric is not None:
                    logs[f"best_{metric_name}"] = best_metric
                    if best_update_number is not None:
                        logs["best_update_number"] = best_update_number
                if not DEBUG:
                    accelerator.log(logs, step=state["global_step"])

            accelerator.wait_for_everyone()

            # If there is no future saving and evaluation for certain, we can skip the rest training.
            if (
                epoch_num == config.train.epoch_num
                and batch_num + (config.train.eval_every * accelerator.gradient_accumulation_steps) > len(dataloaders["train"])
            ):
                break

            # ---- Training step ----
            with accelerator.accumulate(denoiser):
                denoiser.train()
                input_ids: torch.Tensor = batch["data"]
                question_length = batch["question_length"]
                answer_length = batch.get("answer_length", None)
                losses = compute_loss_by_config(
                    input_ids,
                    denoiser,
                    question_length,
                    config=config,
                    answer_length=answer_length,
                )
                loss_tgt = losses["loss"]
                if config.finetuning_method in ["tlora", "nora"]:
                    noise_level = losses["noise_level"]
                torch.cuda.empty_cache()
                accelerator.backward(loss_tgt)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_learn, 1.0)
                optimizer.step()
                scheduler.step()
                    
                optimizer.zero_grad()
                # denoiser.state_dict()['module.base_model.model.model.transformer.blocks.31.q_proj.lora_C.default'].requires_grad
            state["global_step"] += 1
            progress_bar.update(1)

            # ---- Loss  ----
            logged_loss = accelerator.gather(loss_tgt.detach()).mean().item()
            if config.finetuning_method in ["tlora"]:
                logged_noise_level = accelerator.gather(noise_level.detach()).mean().item()
            accum_loss += logged_loss
            accum_count += 1
            # ---- Sample / Token counts  ----
            local_samples = torch.tensor(len(batch["data"]), device=accelerator.device)
            local_tokens = torch.tensor(
                len(batch["data"]) * batch["data"].shape[-1],
                device=accelerator.device,
            )
            total_samples = accelerator.gather(local_samples).sum().item()
            total_tokens = accelerator.gather(local_tokens).sum().item()
            if accelerator.is_main_process:
                state["global_sample_number"] += total_samples
                state["global_token_number"] += total_tokens
            accelerator.wait_for_everyone()

            if accelerator.sync_gradients and accelerator.is_main_process:
                avg_loss_per_update = accum_loss / accum_count
                accum_loss = 0.0
                accum_count = 0

                if not DEBUG:
                    accelerator.log(
                        {"avg_loss_per_update": avg_loss_per_update},
                        step=state["global_step"],
                    )
                state["global_update_number"] += 1


                if config.finetuning_method in ("clora","nara") and stage_1_limit_step > 0:
                    # Switch to Stage 2 exactly when we pass the limit
                    if state["global_update_number"] == stage_1_limit_step:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            print(f"\n[{config.finetuning_method}] Reached step {state['global_update_number']}. Switching to Stage 2 (Training A, B, C, Lambda)...")
                        
                        # Unwrap is necessary to access custom methods on DDP wrapped models
                        unwrapped_model = accelerator.unwrap_model(denoiser)
                        if hasattr(unwrapped_model, "set_training_stage"):
                            unwrapped_model.set_training_stage(2)
                            
            # ---- Logs (only on main process) ----
            progress_bar.set_postfix(
                {
                    "loss": logged_loss,
                    "epoch": epoch_num,
                }
            )
            accelerator.wait_for_everyone()

        # Debug mode: stop after first epoch
        if args.debug:
            if accelerator.is_main_process:
                print(f"[DEBUG] Stopping after epoch {epoch_num} (debug mode)")
            break

    _save_final_ckpt()

    # ---- Experiment Logging: Save results to CSV ----
    if accelerator.is_main_process:
        import csv

        # Calculate training metrics
        total_time_min = (time.time() - training_start_time) / 60
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
        time_per_epoch = total_time_min / config.train.epoch_num

        # Get final validation loss (from last logged value)
        final_val_loss = logs.get("val_loss", None)

        # GPU info
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
        else:
            gpu_name = "N/A"

        # Prepare CSV row
        csv_row = {
            "exp_id": exp_config["exp_id"],
            "method": exp_config["method"],
            "task": exp_config["task"],
            "model": exp_config["model"],
            "lr": exp_config["lr"],
            "seed": exp_config["seed"],
            "rank": exp_config["rank"],
            "epochs": exp_config["epochs"],
            "trainable_params_M": param_stats["total_trainable_M"],
            "total_params_M": param_stats["total_all_M"],
            "trainable_ratio_percent": round(param_stats["trainable_ratio"], 4),
            "A_params_M": param_stats["A_params_M"],
            "B_params_M": param_stats["B_params_M"],
            "gpu_name": gpu_name,
            "peak_vram_gb": round(peak_vram_gb, 2),
            "train_time_min": round(total_time_min, 1),
            "time_per_epoch_min": round(time_per_epoch, 1),
            "best_loss": best_metric,
            "best_step": best_update_number,
            "final_loss": final_val_loss,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        for k, v in csv_row.items():
            print(f"  {k}: {v}")
        print("="*60 + "\n")

        # Save to CSV
        results_dir = config.paths.get("results_dir", "")
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d")
            csv_filename = f"{exp_config['method']}_{exp_config['task']}_lr_ablation_seed_{exp_config['seed']}_{date_str}.csv"
            csv_path = os.path.join(results_dir, csv_filename)

            # Check if file exists to determine if we need header
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_row.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(csv_row)

            print(f"[CSV] Results appended to: {csv_path}")
        else:
            print("[CSV] Warning: paths.results_dir not configured, skipping CSV output")

    accelerator.end_training()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/llada.yaml")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config")
    parser.add_argument("--ckpt_mapping", type=str, default=None, help="Path to ckpt_mapping.py for auto-update after training")
    parser.add_argument("--debug", action="store_true", help="Debug mode: process at most 5 batches then stop")
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
