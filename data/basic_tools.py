from datasets import load_dataset
import torch
from accelerate import Accelerator
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from typing import List
from config import MODEL_PATHS_MAPPING,MODEL_TYPE,get_type,TASK_TYPE,TASK_PATHS_MAPPING
import os
# from config import TASK_TYPE, get_type, , MAX_LENGTH_MAPPING

class CustomDataset(Dataset):
    """Custom dataset class for loading and processing data of LIST type."""

    def __init__(self, data: List):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def process_dataset(
    raw_dataset,
    desc,
    split_name,
    accelerator: Accelerator,
    tokenizer,
    max_length,
    collect_stats,
    stats_dict,
    q_key="instruction",
    a_key="output",
    gt_key=None,
    task_type = None,
):
    """
    Process dataset: tokenize, pad, and optionally collect statistics.

    Args:
        raw_dataset: HuggingFace dataset or list of dicts
        desc: description for tqdm
        split_name: "train" or "val"
        accelerator: accelerator object (used to check main process)
        tokenizer: HuggingFace tokenizer
        max_length: maximum sequence length
        collect_stats: bool, whether to collect statistics
        stats_dict: dict for storing statistics
        q_key: field name for the question (default: "instruction" for commonsense_170k)
        a_key: field name for the answer (default: "output" for commonsense_170k)
        gt_key: field name for the ground truth answer (eg. "answer" for commonsense_170k)
    """
    processed = []
    data_iter = (
        tqdm(raw_dataset, desc=desc, unit="samples")
        if accelerator.is_main_process
        else raw_dataset
    )
    raw_max_length = max_length
    for data in data_iter:
        # Flexible key mapping
        question = data[q_key]
        answer = data[a_key]
        # Tokenize question using chat template

        if gt_key is not None:
            gt = data[gt_key]
        messages = [{"role": "user", "content": question}]
        question: torch.Tensor = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).input_ids[0]

        # Tokenize answer
        answer = tokenizer(answer, return_tensors="pt")["input_ids"][0]
        answer = torch.cat((answer, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        q_len = question.shape[-1]
        a_len = answer.shape[-1]
        total_len = q_len + a_len

        # Skip if too long
        # if task_type in (TASK_TYPE.COMMONSENSE170K,):
        #     max_length = min(raw_max_length, q_len+8)
        
        if total_len > max_length:
            continue

        # Pad to max_length
        padding_length = max_length - total_len
        padding = torch.full(
            (padding_length,), tokenizer.eos_token_id, dtype=question.dtype
        )
        padded_data = torch.cat((question, answer, padding), dim=-1)
        if gt_key is not None:
            processed.append(
                dict(
                    data=padded_data,
                    question_length=q_len,
                    answer_length=a_len,
                    length=total_len,
                    ground_truth=gt,
                )
            )
        else:
            processed.append(
                dict(
                    data=padded_data,
                    question_length=q_len,
                    answer_length=a_len,
                    length=total_len,
                )
            )

        # Collect stats if enabled
        if collect_stats:
            stats_dict[split_name]["q_lens"].append(q_len)
            stats_dict[split_name]["a_lens"].append(a_len)
            stats_dict[split_name]["total_lens"].append(total_len)

    return processed


def print_stats(name, values):
    return (
        f"{name}:\n"
        f"  Mean : {np.mean(values):.2f}\n"
        f"  Std  : {np.std(values):.2f}\n"
        f"  Min  : {np.min(values)}\n"
        f"  Max  : {np.max(values)}\n"
        f"  Count: {len(values)}\n"
    )


def stats(name, arr):
    """Return formatted statistics string for an array."""
    return (
        f"{name} -> "
        f"Mean: {np.mean(arr):.2f}, "
        f"Std: {np.std(arr):.2f}, "
        f"Min: {np.min(arr)}, "
        f"Max: {np.max(arr)}"
    )


def length_analysis(args):
    config = OmegaConf.load(args.config)
    model_type: MODEL_TYPE = get_type(MODEL_TYPE, config.get("model", None))
    if model_type not in MODEL_PATHS_MAPPING.keys():
        raise NotImplementedError(
            f"No path found in MODEL_PATHS_MAPPING for {model_type.__name__}, please specify one."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS_MAPPING[model_type], trust_remote_code=True
    )
    if config.task_name == "commonsense_170k":
        dataset = load_dataset("zwhe99/commonsense_170k", split="train")
    elif config.task_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="train")
    elif config.task_name == "math14k":
        task_type: TASK_TYPE = get_type(TASK_TYPE, config.get("task_name", None))
        task_info = TASK_PATHS_MAPPING[task_type]
        dataset = load_dataset("json", data_files=task_info["data_files"],split="train")
    else:
        raise NotImplementedError(
            f"Unsupported dataset: {config.task_name}, unexpected error may occur."
        )

    # 2. Raw length statistics
    if config.task_name in ["commonsense_170k","math14k"]:
        fields = ["instruction", "answer", "input", "output"]
    elif config.task_name in ["gsm8k"]:
        fields = ["question", "answer"]
    else:
        raise NotImplementedError(
            f"Unsupported dataset: {config.task_name}, fields should be specified."
        )

    raw_lengths = {field: [] for field in fields}

    for sample in tqdm(dataset, desc="Caculating Raw Data...", unit="samples"):
        for field in fields:
            text = sample.get(field, "")
            raw_lengths[field].append(len(text))

    print("=" * 40)
    print("📊 Raw Text Length Analysis (chars)")
    print("=" * 40)
    for field in fields:
        print(print_stats(field.capitalize(), raw_lengths[field]))
    print("=" * 40)

    # 3. Token length statistics
    q_token_lengths, a_token_lengths, total_token_lengths = [], [], []
    
    # Initialize counters for each threshold for question, answer, and total tokens
    q_under_128, q_under_256, q_under_512, q_under_1024 = 0, 0, 0, 0
    a_under_128, a_under_256, a_under_512, a_under_1024 = 0, 0, 0, 0
    total_under_128, total_under_256, total_under_512, total_under_1024 = 0, 0, 0, 0
    
    for data in tqdm(dataset, desc="Processing Data...", unit="samples"):
        if config.task_name in ("commonsense_170k", "math14k"):
            question = data["instruction"]
            answer = data["output"]
        else:
            question = data["question"]
            answer = data["answer"]

        # Convert question
        messages = [{"role": "user", "content": question}]
        question_tokens = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).input_ids[0]

        # Convert answer
        answer_tokens = tokenizer(answer, return_tensors="pt")["input_ids"][0]

        answer_tokens = torch.cat(
            (answer_tokens, torch.tensor([tokenizer.eos_token_id])), dim=-1
        )

        # Append token lengths
        q_token_lengths.append(len(question_tokens))
        a_token_lengths.append(len(answer_tokens))
        total_tokens = len(question_tokens) + len(answer_tokens)
        total_token_lengths.append(total_tokens)

        # Count lengths under each threshold for question
        if len(question_tokens) <= 128:
            q_under_128 += 1
        if len(question_tokens) <= 256:
            q_under_256 += 1
        if len(question_tokens) <= 512:
            q_under_512 += 1
        if len(question_tokens) <= 1024:
            q_under_1024 += 1

        # Count lengths under each threshold for answer
        if len(answer_tokens) <= 128:
            a_under_128 += 1
        if len(answer_tokens) <= 256:
            a_under_256 += 1
        if len(answer_tokens) <= 512:
            a_under_512 += 1
        if len(answer_tokens) <= 1024:
            a_under_1024 += 1

        # Count lengths under each threshold for total tokens (question + answer)
        if total_tokens <= 128:
            total_under_128 += 1
        if total_tokens <= 256:
            total_under_256 += 1
        if total_tokens <= 512:
            total_under_512 += 1
        if total_tokens <= 1024:
            total_under_1024 += 1

    print("=" * 40)
    print("🔢 Tokenized Length Analysis (tokens)")
    print("=" * 40)
    print(print_stats("Question tokens", q_token_lengths))
    print(print_stats("Answer tokens", a_token_lengths))
    print(print_stats("Total tokens", total_token_lengths))
    
    # Print percentage breakdown for each threshold
    print(f"Question Length <= 128: {q_under_128} ({q_under_128 / len(dataset) * 100:.2f}%)")
    print(f"Question Length <= 256: {q_under_256} ({q_under_256 / len(dataset) * 100:.2f}%)")
    print(f"Question Length <= 512: {q_under_512} ({q_under_512 / len(dataset) * 100:.2f}%)")
    print(f"Question Length <= 1024: {q_under_1024} ({q_under_1024 / len(dataset) * 100:.2f}%)")
    
    print(f"Answer Length <= 128: {a_under_128} ({a_under_128 / len(dataset) * 100:.2f}%)")
    print(f"Answer Length <= 256: {a_under_256} ({a_under_256 / len(dataset) * 100:.2f}%)")
    print(f"Answer Length <= 512: {a_under_512} ({a_under_512 / len(dataset) * 100:.2f}%)")
    print(f"Answer Length <= 1024: {a_under_1024} ({a_under_1024 / len(dataset) * 100:.2f}%)")

    # Total token length distribution
    print(f"Total Length <= 128: {total_under_128} ({total_under_128 / len(dataset) * 100:.2f}%)")
    print(f"Total Length <= 256: {total_under_256} ({total_under_256 / len(dataset) * 100:.2f}%)")
    print(f"Total Length <= 512: {total_under_512} ({total_under_512 / len(dataset) * 100:.2f}%)")
    print(f"Total Length <= 1024: {total_under_1024} ({total_under_1024 / len(dataset) * 100:.2f}%)")
    print("=" * 40)





# --- 1. Helper Function: Check Answer Type ---
def check_answer_type(text):
    """
    Classifies text into:
    - 'FLOAT': If it is a pure number (int/float).
    - 'ABC': If it is a single letter option (A, B, C, D, E).
    - 'TEXT': Everything else.
    """
    if not isinstance(text, str):
        text = str(text)
    
    cleaned = text.strip()
    
    # Check for Option (A, B, C, D, E) - Case insensitive
    if len(cleaned) == 1 and cleaned.upper() in ['A', 'B', 'C', 'D', 'E']:
        return "ABC"
    
    # Check for Number (Float/Int)
    try:
        float(cleaned.replace(',', '')) # Handle 1,000
        return "FLOAT"
    except ValueError:
        pass
        
    return "TEXT"

# --- 2. Main Analysis Function ---
def length_analysis_func(model_name, task_name, output_folder, split="train", check_type=False):
    """
    Args:
        model_name (str): The key for the model in MODEL_PATHS_MAPPING.
        task_name (str): The task name to look up via get_type/TASK_PATHS_MAPPING.
        output_folder (str): Folder path to save the analysis file. Filename is auto-generated.
        split (str): Dataset split to load (default: 'train').
        check_type (bool): Whether to check the answer type of the gt_key.
    """
    
    # Generate dynamic filename based on task context
    filename = f"{model_name}_{task_name}_{split}_length_report.txt"
    output_path = os.path.join(output_folder, filename)
    
    # Buffer for logging
    log_buffer = []
    def log_print(msg):
        print(msg)
        log_buffer.append(str(msg))

    # --- Load Model ---
    model_type = get_type(MODEL_TYPE, model_name)
    if model_type not in MODEL_PATHS_MAPPING.keys():
        raise NotImplementedError(
            f"No path found in MODEL_PATHS_MAPPING for {model_type.__name__}"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATHS_MAPPING[model_type], trust_remote_code=True
    )

    # --- Load Dataset (New Logic) ---
    # Get the mapping for the current task
    task_type = get_type(TASK_TYPE, task_name)
    task_info = TASK_PATHS_MAPPING[task_type]
    
    # Determine which split to load (using function parameter)
    split_to_load = split

    if "data_files" in task_info:
        # Check if the split we want is defined
        if split_to_load not in task_info["data_files"]:
            raise ValueError(
                f"Split '{split_to_load}' is not defined in data_files for {task_type}. "
                f"Available splits: {list(task_info['data_files'].keys())}"
            )
        
        dataset = load_dataset(
            "json",  # Specify the json loader
            data_files=task_info["data_files"], # Pass the file paths
            split=split_to_load                 # Select the split
        )
    else:
        # Logic for Hub datasets
        dataset = load_dataset(*(task_info["path"]), split=split_to_load)

    # Assign keys dynamically from task_info
    q_key = task_info["q_key"]
    a_key = task_info["a_key"]
    gt_key = task_info.get("gt_key", None)

    # --- Initialize Containers ---
    # Raw lengths for Question and Output (Model Output/Answer field)
    raw_lengths = {q_key: [], a_key: []}
    
    # Token lengths
    q_token_lengths = []
    a_token_lengths = []
    total_token_lengths = []
    
    # Type counters
    answer_types = {"FLOAT": 0, "ABC": 0, "TEXT": 0}
    
    # Threshold counters
    thresholds = [128, 256, 512, 1024]
    q_under = {t: 0 for t in thresholds}
    a_under = {t: 0 for t in thresholds}
    total_under = {t: 0 for t in thresholds}

    # --- Processing Loop ---
    for data in tqdm(dataset, desc=f"Processing {task_name} ({split})...", unit="samples"):
        
        # 1. Retrieve Text
        question_text = data.get(q_key, "")
        answer_text = data.get(a_key, "") # This is the text we analyze for length
        import pdb; pdb.set_trace()
        # 2. Raw Length Stats
        raw_lengths[q_key].append(len(question_text))
        raw_lengths[a_key].append(len(answer_text))
        
        # 3. Answer Type Checking (Optional on gt_key)
        if check_type and gt_key:
            gt_text = data.get(gt_key, "")
            t_type = check_answer_type(gt_text)
            answer_types[t_type] += 1

        # 4. Tokenization
        # Question: Apply chat template
        messages = [{"role": "user", "content": question_text}]
        q_tokens = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        ).input_ids[0]

        # Answer: Regular tokenization + EOS
        a_tokens = tokenizer(answer_text, return_tensors="pt")["input_ids"][0]
        a_tokens = torch.cat((a_tokens, torch.tensor([tokenizer.eos_token_id])), dim=-1)

        q_len = len(q_tokens)
        a_len = len(a_tokens)
        t_len = q_len + a_len

        q_token_lengths.append(q_len)
        a_token_lengths.append(a_len)
        total_token_lengths.append(t_len)

        # 5. Update Thresholds
        for t in thresholds:
            if q_len <= t: q_under[t] += 1
            if a_len <= t: a_under[t] += 1
            if t_len <= t: total_under[t] += 1

    # --- Reporting ---
    log_print("=" * 40)
    log_print(f"📋 Task: {task_name} | Split: {split}")
    log_print(f"🔑 Keys Mapped -> Q: '{q_key}', A: '{a_key}', GT: '{gt_key}'")
    log_print("=" * 40)

    # Report 1: Type Analysis
    if check_type and gt_key:
        log_print("🧪 Ground Truth Type Analysis (from gt_key)")
        log_print("=" * 40)
        total = len(dataset)
        for k, v in answer_types.items():
            log_print(f"Type [{k}]: {v} ({v / total * 100:.2f}%)")
        log_print("=" * 40)

    # Report 2: Raw Char Lengths
    log_print("📊 Raw Text Length Analysis (chars)")
    log_print("=" * 40)
    # Assuming print_stats is a helper available in your environment
    # If print_stats is not imported, you might need to define it or import it.
    # I'm assuming it's available or you will add it.
    # For safety, I will just use basic stats if print_stats is missing logic, 
    # but adhering to previous structure implies it's expected.
    try:
        log_print(print_stats(f"Question ({q_key})", raw_lengths[q_key]))
        log_print(print_stats(f"Output ({a_key})", raw_lengths[a_key]))
    except NameError:
        log_print("Note: print_stats function not found. Skipping detailed stats format.")
        log_print(f"Avg Q Len: {sum(raw_lengths[q_key])/len(raw_lengths[q_key]):.2f}")
        log_print(f"Avg A Len: {sum(raw_lengths[a_key])/len(raw_lengths[a_key]):.2f}")

    log_print("=" * 40)

    # Report 3: Token Lengths
    log_print("🔢 Tokenized Length Analysis (tokens)")
    log_print("=" * 40)
    try:
        log_print(print_stats("Question Tokens", q_token_lengths))
        log_print(print_stats("Answer Tokens", a_token_lengths))
        log_print(print_stats("Total Tokens", total_token_lengths))
    except NameError:
        pass

    # Report 4: Threshold Distribution
    log_print("-" * 20)
    total_samples = len(dataset)
    
    log_print(f"--- Question Tokens ({q_key}) ---")
    for t in thresholds:
        log_print(f"<= {t}: {q_under[t]} ({q_under[t] / total_samples * 100:.2f}%)")
    
    log_print(f"\n--- Answer Tokens ({a_key}) ---")
    for t in thresholds:
        log_print(f"<= {t}: {a_under[t]} ({a_under[t] / total_samples * 100:.2f}%)")
        
    log_print("\n--- Total Tokens ---")
    for t in thresholds:
        log_print(f"<= {t}: {total_under[t]} ({total_under[t] / total_samples * 100:.2f}%)")
    
    log_print("=" * 40)

    # --- Save Output ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(log_buffer))
        print(f"\n✅ Analysis saved to: {output_path}")
    except Exception as e:
        print(f"\n❌ Failed to save output file: {e}")