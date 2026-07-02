from transformers import AutoModel, AutoTokenizer
from peft import get_peft_model, PeftModel
import torch
from omegaconf import OmegaConf
from config import MODEL_TYPE, FINETUNING_TYPE, get_type, MODEL_PATHS_MAPPING


def get_model_by_config(config):
    """Select model based on config file"""
    model_type: MODEL_TYPE = get_type(MODEL_TYPE, config.get("model", None))
    finetuning_type: FINETUNING_TYPE = get_type(
        FINETUNING_TYPE, config.get("finetuning_method", None)
    )
    if model_type not in MODEL_PATHS_MAPPING.keys():
        raise NotImplementedError(
            f"No path found in MODEL_PATHS_MAPPING for {model_type.__name__}, please specify one."
        )

    if finetuning_type is FINETUNING_TYPE.NARA:
        return get_nara_models(model_type, config)
    else:
        raise NotImplementedError(
            f"Unsupported finetuning method: {finetuning_type}"
        )


def _get_resume_path(config):
    """Helper to extract resume path safely"""
    if hasattr(config, "train") and hasattr(config.train, "decoder_resume_path"):
        val = config.train.decoder_resume_path
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None


def _freeze_base_model(model):
    """Helper to freeze model parameters"""
    for param in model.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)


def get_nara_models(model_type, config):
    if config.data.batch_size != 1:
        raise ValueError("NARA currently only supports batch_size=1.")

    from nara import PeftModel as NARAPeftModel, NARAConfig, get_peft_model as get_nara_peft_model

    # base_model = AutoModel.from_pretrained(
    #     MODEL_PATHS_MAPPING[model_type], trust_remote_code=True
    # )
    base_model = AutoModel.from_pretrained(
        MODEL_PATHS_MAPPING[model_type],
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS_MAPPING[model_type], trust_remote_code=True
    )

    decoder_resume_path = _get_resume_path(config)
    _freeze_base_model(base_model)

    if decoder_resume_path:
        model = NARAPeftModel.from_pretrained(
            base_model, decoder_resume_path, is_trainable=True
        )
        print(f"[NARA] Resumed from: {decoder_resume_path}")
    else:
        ft_params = OmegaConf.to_container(config.finetuning_parameters, resolve=True)
        lora_ckpt_path = ft_params.pop("lora_ckpt_path", None)
        peft_config = NARAConfig(**ft_params)

        if "ff_out" in ft_params["target_modules"]:
            tr = base_model.model.transformer
            assert "ff_out" in tr._modules, "no transformer.ff_out found"
            tr._modules["lm_head_ff_out_tmp"] = tr._modules.pop("ff_out")
            try:
                model = get_nara_peft_model(base_model, peft_config)
            finally:
                tr._modules["ff_out"] = tr._modules.pop("lm_head_ff_out_tmp")
        else:
            model = get_nara_peft_model(base_model, peft_config)

        if lora_ckpt_path:
            print(f"[NARA] Loading warm-start LoRA weights from: {lora_ckpt_path}")
            if hasattr(model, "load_lora_only"):
                model.load_lora_only(lora_ckpt_path)
            elif hasattr(model.base_model, "load_lora_only"):
                model.base_model.load_lora_only(lora_ckpt_path)
            else:
                print("[NARA] Warning: could not find load_lora_only method on model.")

    model.print_trainable_parameters()
    return model, tokenizer
