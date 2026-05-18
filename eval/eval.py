import torch
import numpy as np
import torch.nn.functional as F
from typing import Dict, Any
import re
from tqdm import tqdm
from fraction import Fraction
import math
from accelerate import Accelerator

from config import MODEL_TYPE, get_type, TASK_TYPE, MASK_ID_MAPPING, FINETUNING_TYPE
from .llada_generate import generate


def extract_pred_from_text_commonsense_170k(gold_text: str, answer_decoded: str) -> str:
    """
    Extract predicted answer from answer_decoded based on the format of gold_text.
    """
    gold_text = gold_text.strip().lower()
    sentence_ = answer_decoded.strip().lower()

    if gold_text in ["true", "false"]:  # boolq
        matches = re.findall(r"true|false", sentence_)
    elif gold_text in ["solution1", "solution2"]:  # piqa
        matches = re.findall(r"solution1|solution2", sentence_)
    elif gold_text in [
        "answer1",
        "answer2",
        "answer3",
        "answer4",
        "answer5",
    ]:  # siqa/arcc/arce/obqa
        matches = re.findall(r"answer[1-5]", sentence_)
    elif gold_text in ["ending1", "ending2", "ending3", "ending4"]:  # hellaswag
        matches = re.findall(r"ending[1-4]", sentence_)
    elif gold_text in ["option1", "option2"]:  # winogrande
        matches = re.findall(r"option[1-2]", sentence_)
    else:
        # Default: check if gold_text appears directly in the output
        matches = re.findall(re.escape(gold_text), sentence_)

    if not matches:
        return ""
    return matches[0]

def extract_number(answer_decoded: str) -> str:
    
    sentence = answer_decoded.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    pred_answer = float(pred[-1])
    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer

def extract_last_option(answer_decoded: str) -> str:
    """
    Extracts the last occurrence of A, B, C, or D from the text.
    
    Logic:
    1. It searches for uppercase A, B, C, D surrounded by word boundaries 
       (e.g., matches "A", "(A)", "A.", but not "Apple").
    2. It returns the LAST match found (common in CoT where the answer is at the end).
    3. Returns empty string if no match found.
    """
    if not answer_decoded:
        return ""
        
    # regex explanation:
    # \b      : boundary (ensures we don't match 'A' inside 'Apple')
    # [A-D]   : character set A, B, C, or D
    # \b      : boundary
    # We strictly look for Uppercase A-D because 'a' is a common article in English.
    matches = re.findall(r'\b[A-D]\b', answer_decoded)

    if matches:
        return matches[-1] # Return the last one found
    return ""

def evaluate_per_sample(
    config, accelerator,sample, model, tokenizer, evaluating_bar, total, correct
):
    """
    Evaluate a single sample from the commonsense_170k dataset.

    Args:
        config: Configuration object containing evaluation parameters.
        sample: A dictionary with keys "question_length", "data", and "ground_truth".
        model: The model to run generation with.
        tokenizer: Tokenizer used for decoding generated tokens.
        evaluating_bar: Progress bar object for evaluation.
        total: Current count of evaluated samples.
        correct: Current count of correct predictions.

    Returns:
        total (int): Updated total number of evaluated samples.
        correct (int): Updated number of correct predictions.
    """
    # 1) Extract question length
    question_length = sample["question_length"][-1]

    # Input prompt tokens: only take the question portion
    prompt_ids: torch.Tensor = sample["data"][:, :question_length]

    # Ground truth (gold answer) string
    gold_text: str = sample["ground_truth"][0]

    # -------------------------------
    # 2) Run llada_generate
    # -------------------------------
    # Generate predicted tokens given the prompt.
    finetuning_type: FINETUNING_TYPE = get_type(FINETUNING_TYPE, config.get("finetuning_method", None))
    mask_id = MASK_ID_MAPPING[MODEL_TYPE(config.get("model", None))]
    gen_tokens: torch.Tensor = generate(
        model=model,
        finetuning_type=finetuning_type,
        direct_noise=True,
        prompt=prompt_ids,
        steps=config.train.eval.steps,
        gen_length=config.train.eval.gen_length,
        block_length=config.train.eval.block_length,
        temperature=config.train.eval.temperature,
        cfg_scale=config.train.eval.cfg_scale,
        remasking=config.train.eval.remasking,
        mask_id=mask_id,
        is_main_process=accelerator.is_main_process,
    )  # shape: [1, prompt_length + gen_length]
    # -------------------------------
    # 3) Decode model output
    # -------------------------------
    # Slice out only the generated part (after the question tokens).
    answer_decoded = tokenizer.batch_decode(
        gen_tokens[:, question_length:], skip_special_tokens=True
    )[0]

    # -------------------------------
    # 4) Extract predicted answer
    # -------------------------------
    task_type: TASK_TYPE = get_type(TASK_TYPE, config.get("task_name", None))
    if task_type in (TASK_TYPE.COMMONSENSE170K,):
        extracted_answer = extract_pred_from_text_commonsense_170k(
            gold_text, answer_decoded
        )
    
    elif task_type in (TASK_TYPE.MATH14K,):
        extracted_answer = extract_number(answer_decoded)
        gold_text=float(gold_text)
    else:
        raise NotImplementedError(f"Task type {task_type} not supported.")
    # -------------------------------
    # 5) Update counters
    # -------------------------------
    total += 1
    correct += (extracted_answer == gold_text)
    evaluating_bar.update(1)
    evaluating_bar.set_postfix(
        {
            "acc_on_main": f"{correct}/{total}={correct / total:.2f}",
        }
    )

    return total, correct

def evaluate_llada(
    accelerator: Accelerator,
    dataloader,
    model,
    tokenizer,
    config,
) -> Dict[str, Any]:
    """
    Evaluation framework for llada_generate:

    Workflow:
    1. Iterate through the evaluation dataloader.
    2. For each sample:
       - Extract the input prompt and gold (ground truth) answer.
       - Use llada_generate to produce a sequence prediction from the model.
       - Decode the generated tokens into text.
       - Extract the predicted answer from the decoded text.
       - Compare the prediction with the ground truth.
    3. Collect evaluation metrics such as accuracy.

    Args:
        accelerator: HuggingFace Accelerate object for distributed setup (controls progress bar visibility).
        dataloader: DataLoader yielding evaluation samples with "data", "question_length", and "ground_truth".
        model: The trained model used for generation.
        tokenizer: Tokenizer used for decoding the model outputs.
        config: Configuration object with generation parameters (steps, gen_length, etc.).

    Returns:
        val_metrics: A dictionary containing:
            - accuracy: Ratio of correct predictions to total samples.
    """
    total = 0
    correct = 0

    # Only show progress bar on the main process (avoids multiple bars in distributed training).
    # dataloader = tqdm(dataloader, desc="Evaluating", unit="samples") if accelerator.is_main_process else dataloader
    evaluating_bar = tqdm(
        total=len(dataloader),
        initial=0,
        desc="Evaluating",
        unit="samples",
        leave=False,
        disable=not accelerator.is_local_main_process,
    )
    for sample in dataloader:
        if TASK_TYPE(config.task_name) in (TASK_TYPE.COMMONSENSE170K,TASK_TYPE.MATH14K
                                           ):
            total, correct = evaluate_per_sample(
                config,
                accelerator,
                sample,
                model,
                tokenizer,
                evaluating_bar,
                total,
                correct,
            )
        else:
            raise NotImplementedError(f"Unsupported dataset: {config.data.name}")

    # -------------------------------
    # 6) Compute final metrics
    # -------------------------------
    val_metrics = {
        "num_samples": torch.tensor(total, device=accelerator.device),
        "num_correct": torch.tensor(correct, device=accelerator.device),
    }
    return val_metrics


def evaluate_model(
    accelerator: Accelerator,
    dataloader,
    model,
    tokenizer,
    config,
) -> Dict[str, Any]:
    model_type: MODEL_TYPE = get_type(MODEL_TYPE, config.get("model", None))
    if model_type in (
        MODEL_TYPE.LLADA_INSTRUCT,
    ):
        return evaluate_llada(accelerator, dataloader, model, tokenizer, config)
    else:
        raise NotImplementedError(f"Unsupported model: {config.model} for evaluation.")
