import os
import re
import torch
from transformers import AutoModel, AutoTokenizer
from config import FINETUNING_TYPE


BASE_MODEL_PATHS = {
    "llada_instruct": "GSAI-ML/LLaDA-8B-Instruct",
}


def _find_best_checkpoint(ft_task="math14k", rank=32, lr=1e-4, c_scale=0.1, stage_1=0):
    exp_root = "experiments"
    if not os.path.isdir(exp_root):
        raise FileNotFoundError(f"experiments directory not found: {exp_root}")

    candidates = []
    for name in os.listdir(exp_root):
        if ft_task not in name:
            continue
        if "nara" not in name:
            continue
        ckpt_dir = os.path.join(exp_root, name, "ckpts")
        if not os.path.isdir(ckpt_dir):
            continue

        for ckpt_name in os.listdir(ckpt_dir):
            path = os.path.join(ckpt_dir, ckpt_name)
            if os.path.isdir(path) and ckpt_name.startswith("BEST_loss_"):
                m = re.search(r"BEST_loss_([0-9.]+)", ckpt_name)
                loss = float(m.group(1)) if m else 999999.0
                candidates.append((loss, path))

    if not candidates:
        raise FileNotFoundError(
            "No BEST_loss_* checkpoint found under experiments/*/ckpts. "
            "Please check your checkpoint path."
        )

    candidates.sort(key=lambda x: x[0])
    best = candidates[0][1]
    print(f"[EVAL] Using checkpoint: {best}")
    return best


def _freeze_base_model(model):
    for p in model.parameters():
        p.requires_grad = False
        if p.ndim == 1:
            p.data = p.data.to(torch.float32)


def get_eval_model(
    base_model_name,
    peft_name=None,
    ft_task="math14k",
    run_time=1,
    f_form="linear",
    zero_lora_init=False,
    direct_noise=False,
    ckpts="best",
    training_mode="joint",
    t_mapping="poly",
    fnn_hidden_size=32,
    lr=1e-4,
    use_embedding=True,
    embedding_dim=64,
    init_c="zero_last",
    density_radius=5,
    rank=32,
    fnn_hidden_size_2=512,
    Embed_components="nl",
    Embed_type="fourier",
    c_scale=0.1,
    stage_1=0,
    scale_ab=1.0,
    clr=1e-4,
    ablation_mode=None,
    ablation_lr=None,
    ablation_seed=None,
):
    if base_model_name not in BASE_MODEL_PATHS:
        raise ValueError(f"Unsupported base_model_name: {base_model_name}")

    model_path = BASE_MODEL_PATHS[base_model_name]

    print(f"[EVAL] Loading base model: {model_path}")
    base_model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if peft_name in (None, "", "none"):
        return base_model, tokenizer, None

    if peft_name != "nara":
        raise ValueError(f"This minimal eval/utils.py only supports peft_name='nara', got: {peft_name}")

    from nara import PeftModel as NARAPeftModel

    _freeze_base_model(base_model)

    if ckpts == "best":
        ckpt_path = _find_best_checkpoint(
            ft_task=ft_task,
            rank=rank,
            lr=lr,
            c_scale=c_scale,
            stage_1=stage_1,
        )
    else:
        ckpt_path = ckpts

    print(f"[EVAL] Loading NaRA checkpoint: {ckpt_path}")
    model = NARAPeftModel.from_pretrained(
        base_model,
        ckpt_path,
        is_trainable=False,
    )

    return model, tokenizer, FINETUNING_TYPE.NARA
