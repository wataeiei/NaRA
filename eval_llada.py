"""
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/hf_cache"

import accelerate
import torch
import torch.nn.functional as F
from datasets import Dataset


import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from functools import partial
import random
import numpy as np
from eval.llada_generate import generate
from eval.utils import get_eval_model


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist_peft")
class LLaDAEvalHarnessLora(LM):
    def __init__(
        self,
        base_model_name,  # original backbone model
        peft_name=None,  # path to your LoRA checkpoint folder (Ckpt_xxx). None/"" = no LoRA
        ft_task="gsm8k",
        run_time=1,  # number of times to run eval (for multi-gpu with small num of samples)
        f_form="linear",
        training_mode="joint",
        t_mapping="poly",
        fnn_hidden_size = 32,
        lr=5e-4 ,
        direct_noise=False,
        ckpts="best",
        zero_lora_init=False,
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=True,
        cfg=0.0,
        steps=256,
        gen_length=512,
        block_length=512,
        remasking="low_confidence",
        device="cuda",
        random_noise=False,
        till_eos=True,
        embedding_dim=64,
        density_radius=5,
        init_c="zero_last",
        rank=32,
        fnn_hidden_size_2=512,
        Embed_components="nl",
        Embed_type="fourier",
        c_scale=1.0,
        stage_1=0,
        scale_ab=1.0,
        clr=1e-4,
        ablation_mode=None,
        ablation_lr=None,
        ablation_seed=None,
        debug=False,  # Debug mode: print noise_level before each forward
        **kwargs,
    ):
        super().__init__()
        accelerator = accelerate.Accelerator()

        self.accelerator = accelerator
        self.debug = debug  # Store debug flag

        # Load base model
        self.direct_noise = direct_noise
        self.random_noise = random_noise
        self.till_eos = till_eos

        # Normalize ablation_lr to string format (lm-eval parses 5e-5 as float 5e-05)
        ablation_lr_normalized = ablation_lr
        if ablation_lr is not None and isinstance(ablation_lr, float):
            if ablation_lr == 5e-5:
                ablation_lr_normalized = "5e-5"
            elif ablation_lr == 1e-4:
                ablation_lr_normalized = "1e-4"
            elif ablation_lr == 2e-4:
                ablation_lr_normalized = "2e-4"
            else:
                ablation_lr_normalized = str(ablation_lr)
        ablation_seed_normalized = int(ablation_seed) if ablation_seed is not None else None

        self.model, self.tokenizer, self.finetuning_type = get_eval_model(
            base_model_name,
            peft_name,
            ft_task,
            run_time,
            f_form,
            zero_lora_init,
            direct_noise,
            ckpts,
            training_mode,
            t_mapping,
            fnn_hidden_size,
            lr,
            use_embedding=True,
            embedding_dim=embedding_dim,
            init_c=init_c,
            density_radius=density_radius,
            rank=rank,
            fnn_hidden_size_2=fnn_hidden_size_2,
            Embed_components=Embed_components,
            Embed_type=Embed_type,
            c_scale=c_scale,
            stage_1=stage_1,
            scale_ab=scale_ab,
            clr=clr,
            ablation_mode=ablation_mode,
            ablation_lr=ablation_lr_normalized,
            ablation_seed=ablation_seed_normalized,
        )
        if peft_name in ("tlora","nora"):
            real_model = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            model_direct_noise = real_model.peft_config["default"].direct_noise_level
            if model_direct_noise != direct_noise:
                raise ValueError(
                    f"Provided direct_noise {direct_noise} inconsistent with that in the model {model_direct_noise}"
                )
        # self.model.state_dict()['base_model.model.model.transformer.blocks.31.q_proj.lora_A.default.weight']
        self.model.eval()
        # for name, param in self.model.named_parameters():
        #     if "lora_" in name:
        #         print("Before prepare",name, param.dtype)
        # self.check_dtype(self.model, tag="before prepare")

        # Enable debug mode for DoRA_V2 if debug flag is set
        if self.debug:
            real_model = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(real_model, "set_debug_mode"):
                real_model.set_debug_mode(True)
                print("[DEBUG] DoRA_V2 debug mode enabled - will print noise_level before each forward")

        self.device = torch.device(device)
        # self.model = self.accelerator.prepare(self.model)
        
        self.device = torch.device(f"{self.accelerator.device}")
        self.model = self.model.to(accelerator.device)
        self._rank = self.accelerator.local_process_index
        self._world_size = self.accelerator.num_processes

        # _ = input("Press Enter to continue...")

        self.mask_id = mask_id

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _verify_equivalence(self, base_model, lora_model, device="cuda"):
        import torch

        base_model.eval()
        lora_model.eval()

        # Dummy input for comparison
        input_ids = torch.randint(
            0, base_model.config.vocab_size, (2, 16), device=device
        )
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
            out_base = base_model(input_ids).logits
            out_lora = lora_model(input_ids).logits

        # print("out_base dtype:", out_base.dtype)
        # print("out_lora dtype:", out_lora.dtype)
        out_base = out_base.to(torch.bfloat16)
        out_lora = out_lora.to(torch.bfloat16)
        is_equal = torch.allclose(out_base, out_lora, atol=1e-6, rtol=1e-5)

        diff = torch.abs(out_base - out_lora).max().item()

        print(
            f"[LoRA Zero-Init Check] Equal within tolerance: {is_equal}, max diff = {diff:.6e}"
        )

    def check_dtype(self, model, tag):
        for name, param in model.named_parameters():
            print(f"[{tag}] {name}: {param.dtype}")
            break

    def _forward_process(self, batch, prompt_index):
        raise NotImplementedError
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b, prompt_index.sum(), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        raise NotImplementedError
        if self.cfg > 0.0:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.0:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        raise NotImplementedError
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_id, device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    @staticmethod
    def loglikelihood_tokenize(example, encode_pair):
        prefix, target = encode_pair(example["prefix"], example["target"])
        return {
            "prefix_text": example["prefix"],
            "target_text": example["target"],
            "prefix": prefix,
            "target": target,
        }

    @staticmethod
    def generate_until_tokenize(example, tokenizer):
        return {
            "question": tokenizer(example["question"])["input_ids"],
            "question_text": example["question"],
            "until": example["until"],
        }

    def loglikelihood(self, requests):
        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(partial(self.loglikelihood_tokenize, encode_pair=self._encode_pair))
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]  # type: ignore

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]  # type: ignore
                target = elem["target"]  # type: ignore

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests: list[Instance]):
        ds = [
            {"question": req.args[0], "until": req.args[1]["until"]} for req in requests
        ]  # type: ignore
        ds = Dataset.from_list(ds)
        ds = ds.map(partial(self.generate_until_tokenize, tokenizer=self.tokenizer))
        ds = ds.with_format("torch")

        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)  # type: ignore
            stop_tokens = elem["until"]  # type: ignore

            generated_answer = generate(
                self.model,
                self.tokenizer,
                self.finetuning_type,
                self.direct_noise,
                prompt,
                steps=self.steps,
                gen_length=self.gen_length,
                block_length=self.block_length,
                temperature=0,
                cfg_scale=self.cfg,
                remasking=self.remasking,
                is_main_process=self.accelerator.is_main_process,
                mask_id=self.mask_id,
                random_noise=self.random_noise,
                till_eos=self.till_eos,
            )

            generated_answer = self.tokenizer.decode(
                generated_answer[0][prompt.shape[1] :], skip_special_tokens=False
            )
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)

            self.accelerator and self.accelerator.wait_for_everyone()  # type: ignore

        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
