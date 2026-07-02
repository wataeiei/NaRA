"""
Step 1: 验证 dLLM 层重要性的双峰规律
对 LLaDA + NaRA 计算相邻层 hidden state 的余弦相似度
显存优化版：FP16 + 逐噪声处理 + 及时释放
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/root/hf_cache"

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
import sys, json, random, gc

# 配置
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"
# 改为你的 NaRA checkpoint 路径
CKPT_PATH = "/mnt/NaRA/experiments/llada_instruct_nara_math14k_r_32_lr_0.0001_epc_10_cscale_0.1_stage1_0/ckpts/BEST_loss_0.049770_seed_1234_update_1472_epoch_4"
# 如果只想跑原始 LLaDA 不加 NaRA，把下面这行注释掉
# CKPT_PATH = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "bimodal_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 清理显存
torch.cuda.empty_cache()
gc.collect()

# ---------- 1. 加载模型（FP16）----------
print(f"Loading LLaDA from {MODEL_NAME} ...")
base_model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,       # FP16 加载
    low_cpu_mem_usage=True,
).to(DEVICE)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

if CKPT_PATH is not None and os.path.exists(CKPT_PATH):
    from nara import PeftModel as NARAPeftModel
    model = NARAPeftModel.from_pretrained(base_model, CKPT_PATH, is_trainable=False)
    print(f"Loaded NaRA checkpoint from {CKPT_PATH}")
else:
    if CKPT_PATH is not None:
        print(f"CKPT_PATH {CKPT_PATH} not found, using raw LLaDA")
    model = base_model

model = model.to(DEVICE)
model.eval()
print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GiB")

# ---------- 2. Hook ----------
hidden_states = {}

def get_hook(layer_idx):
    def hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        hidden_states[layer_idx] = h.detach()
    return hook

# 找到 transformer blocks（LLaDA 结构）
# NaRA PeftModel -> base_model (LLaDAModelLM) -> .model (LLaDAModel) -> .transformer.blocks
if hasattr(model, 'base_model'):
    raw_llada = model.base_model  # LLaDAModelLM
elif hasattr(model, 'model'):
    raw_llada = model.model
else:
    raw_llada = model

print(f"Model type: {type(raw_llada).__name__}")
raw_model = raw_llada.model  # LLaDAModel

if hasattr(raw_model, 'transformer'):
    trans = raw_model.transformer
    if hasattr(trans, 'blocks'):
        blocks = trans.blocks
    elif isinstance(trans, dict):
        blocks = trans['blocks']
    else:
        blocks = trans['blocks'] if isinstance(trans, (dict,)) else None
elif hasattr(raw_model, 'blocks'):
    blocks = raw_model.blocks
else:
    raise AttributeError(f"Cannot find blocks. Check model structure.")

print(f"Found {len(blocks)} transformer blocks")

num_layers = len(blocks)
print(f"Registering hooks on {num_layers} transformer blocks ...")

# ---------- 3. 输入数据 ----------
test_texts = [
    "Question: Tom has 3 apples. He buys 2 more. How many apples does he have now?\nAnswer:",
    "Question: A train travels 120 miles in 2 hours. What is its average speed?\nAnswer:",
    "Question: If x + 5 = 12, what is x?\nAnswer:",
]
prompt_len = 30   # 短 prompt
answer_len = 24   # 短 response
seq_len = prompt_len + answer_len

print(f"Sequence length: prompt={prompt_len}, answer={answer_len}, total={seq_len}")

# ---------- 4. 逐噪声水平处理 ----------
all_results = {}
all_similarities = []

for noise_level in [0.25, 0.50, 0.75, 0.95]:
    print(f"\n{'='*40}")
    print(f"Processing noise_level λ={noise_level} ...")

    layer_sims_list = []

    for text in test_texts:
        hidden_states.clear()
        gc.collect()

        # Tokenize + 构造输入
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                           max_length=prompt_len, truncation=True).to(DEVICE)
        input_ids = inputs["input_ids"]

        # 构造完整序列：prompt + mask response
        full_ids = torch.full((1, seq_len), tokenizer.mask_token_id,
                              dtype=torch.long, device=DEVICE)
        full_ids[:, :prompt_len] = input_ids[:, :prompt_len]

        # 注册 hook
        handles = []
        for i, block in enumerate(blocks):
            handles.append(block.register_forward_hook(get_hook(i)))

        # Forward
        with torch.no_grad():
            _ = model(full_ids)

        # 清理 hook
        for h in handles:
            h.remove()

        # 计算余弦相似度
        layer_sims = []
        keys = sorted(hidden_states.keys())
        for i in range(1, len(keys)):
            h_prev = hidden_states[keys[i-1]].float()  # (1, seq_len, dim)
            h_curr = hidden_states[keys[i]].float()

            # 只取 response 部分
            h_prev_r = h_prev[:, prompt_len:, :]
            h_curr_r = h_curr[:, prompt_len:, :]

            # 归一化
            h_prev_n = h_prev_r / (h_prev_r.norm(dim=-1, keepdim=True) + 1e-8)
            h_curr_n = h_curr_r / (h_curr_r.norm(dim=-1, keepdim=True) + 1e-8)

            cos_sim = (h_prev_n * h_curr_n).sum(dim=-1).mean().item()
            layer_sims.append(cos_sim)

        layer_sims_list.append(layer_sims)

        # 及时释放
        del full_ids, inputs, _
        gc.collect()
        torch.cuda.empty_cache()

    # 对多个文本取平均
    avg_sims = np.mean(layer_sims_list, axis=0)
    all_similarities.append(avg_sims)
    all_results[f"lambda_{noise_level}"] = {
        k: float(v) for k, v in enumerate(avg_sims)
    }
    print(f"  Done. Sample layer sims: {[f'{s:.4f}' for s in avg_sims[:5]]} ...")

# ---------- 5. 绘图 ----------
plt.figure(figsize=(10, 5))
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
for idx, (sims, nl) in enumerate(zip(all_similarities, [0.25, 0.50, 0.75, 0.95])):
    plt.plot(range(1, len(sims)+1), sims, label=f'λ={nl}',
             color=colors[idx], linewidth=2, alpha=0.8)

plt.axvspan(0, 8, alpha=0.08, color='blue', label='Front (0-7)')
plt.axvspan(24, 32, alpha=0.08, color='red', label='Back (24-31)')
plt.axvspan(8, 24, alpha=0.08, color='gray', label='Middle (8-23)')

plt.xlabel('Layer Index', fontsize=12)
plt.ylabel('Cosine Similarity (adjacent layers)', fontsize=12)
plt.title(f'Bimodal Distribution: Layer Importance\n{model.__class__.__name__}', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'bimodal_analysis.png'), dpi=150)
print(f"\nFigure saved to {OUTPUT_DIR}/bimodal_analysis.png")

# ---------- 6. 分析结果 ----------
avg_all_noise = np.mean(all_similarities, axis=0)
front_avg = float(np.mean(avg_all_noise[0:7]))
middle_avg = float(np.mean(avg_all_noise[8:23]))
back_avg = float(np.mean(avg_all_noise[24:31]))
bimodal = middle_avg > front_avg and middle_avg > back_avg

print("\n" + "="*50)
print("BIMODAL ANALYSIS RESULTS")
print("="*50)
print(f"  Front layers (0-7)   avg cosine sim: {front_avg:.4f}")
print(f"  Middle layers (8-23) avg cosine sim: {middle_avg:.4f}")
print(f"  Back layers (24-31)  avg cosine sim: {back_avg:.4f}")
print(f"  {'✓ Bimodal CONFIRMED!' if bimodal else '✗ Pattern unclear'}")
print("  (lower cosine sim = more transformation = higher importance)")

results = {
    "ckpt": CKPT_PATH,
    "bimodal_confirmed": bimodal,
    "front_avg_similarity": front_avg,
    "middle_avg_similarity": middle_avg,
    "back_avg_similarity": back_avg,
    "layer_similarities": {str(k): float(v) for k, v in enumerate(avg_all_noise)},
}
with open(os.path.join(OUTPUT_DIR, 'bimodal_results.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {OUTPUT_DIR}/bimodal_results.json")
