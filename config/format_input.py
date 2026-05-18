from enum import Enum
from typing import Union


class TASK_TYPE(Enum):
    GSM8K = "gsm8k"

    
    COMMONSENSE170K = "commonsense170k"
    MATH14K = "math14k"
    MATH50K = "math50k"
    
    #This is for commonsense test sets
    ARC_CHALLENGE = "arc_challenge"
    ARC_EASY = "arc_easy"
    BOOLQ = "boolq"
    HELLASWAG = "hellaswag"
    OPENBOOKQA = "openbookqa"
    PIQA = "piqa"
    SOCIAL_I_QA = "social_i_qa"
    WINOGRANDE = "winogrande"
    
    #This is for math test sets
    ADDSUB = "addsub"
    AQUA = "aqua"
    GSM8K_TEST = "gsm8k_test"
    MULTIARITH = "multiarith"
    SVAMP = "svamp"
    SINGLEEQ = "singleeq"
    MINERVA_MATH = "minerva_math"
    CODE_FEEDBACK = "code_feedback"
    
class MODEL_TYPE(Enum):
    LLADA_INSTRUCT = "llada_instruct"
    LLADA_BASE = "llada_base"
    # DREAM = "dream"


class FINETUNING_TYPE(Enum):
    NARA = "nara"


def get_type(type: Union[TASK_TYPE, MODEL_TYPE, FINETUNING_TYPE], value: str):
    """Check if the value in the config file is valid"""
    try:
        return type(value)
    except ValueError:
        raise ValueError(
            f"Unsupported value: *{value}* for type: *{type.__name__}*, optional values: *{[e.value for e in type]}*"
        )


MODEL_PATHS_MAPPING = {
    MODEL_TYPE.LLADA_INSTRUCT: "GSAI-ML/LLaDA-8B-Instruct",
    MODEL_TYPE.LLADA_BASE: "GSAI-ML/LLaDA-8B-Base",
    # MODEL_TYPE.DREAM: "/path/to/dream/model", # TODO: add dream model path
}
TASK_PATHS_MAPPING = {
    TASK_TYPE.GSM8K: {
        "path": ("openai/gsm8k", "main"),
        "q_key": "question",
        "a_key": "answer",
    },
    TASK_TYPE.GSM8K_TEST: {
        "data_files": {
            "test": "data/llm_adapt/gsm8k/test.json",
            # "test": "data/llm_adapt/commonsense_170k/test.json" # <-- Add this path if you have a test file
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.COMMONSENSE170K: {

        "data_files": {
            "train": "data/llm_adapt/commonsense_170k/train.json",
            # "test": "data/llm_adapt/commonsense_170k/test.json" # <-- Add this path if you have a test file
        },
        "q_key": "instruction",
        "a_key": "output",
    },
    TASK_TYPE.MATH14K: {

        "data_files": {
            "train": "data/llm_adapt/math/math_14k.json",
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.MATH50K: {

        "data_files": {
            "train": "data/llm_adapt/math/math_50k.json",
        },
        "q_key": "instruction",
        "a_key": "output",
    },
    TASK_TYPE.ARC_CHALLENGE: {
        "data_files": {
            "test": "data/llm_adapt/ARC-Challenge/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.ARC_EASY: {
        "data_files": {
            "test": "data/llm_adapt/ARC-Easy/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.BOOLQ: {
        "data_files": {
            "test": "data/llm_adapt/boolq/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.HELLASWAG: {
        "data_files": {
            "test": "data/llm_adapt/hellaswag/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.OPENBOOKQA: {
        "data_files": {
            "test": "data/llm_adapt/openbookqa/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.PIQA: {
        "data_files": {
            "test": "data/llm_adapt/piqa/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.SOCIAL_I_QA: {
        "data_files": {
            "test": "data/llm_adapt/social_i_qa/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.WINOGRANDE: {
        "data_files": {
            "test": "data/llm_adapt/winogrande/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.ADDSUB: {
        "data_files": {
            "test": "data/llm_adapt/AddSub/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.AQUA: {
        "data_files": {
            "test": "data/llm_adapt/AQuA/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.MULTIARITH: {
        "data_files": {
            "test": "data/llm_adapt/MultiArith/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.SVAMP: {
        "data_files": {
            "test": "data/llm_adapt/SVAMP/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.SINGLEEQ: {
        "data_files": {
            "test": "data/llm_adapt/SingleEq/test.json"
        },
        "q_key": "instruction",
        "a_key": "output",
        "gt_key": "answer",
    },
    TASK_TYPE.CODE_FEEDBACK: {
        "data_files": {
            "train": "data/pissa_dataset_python_train/python/train.json"
        },
        "q_key": "instruction",
        "a_key": "output",
    },
}

MAX_LENGTH_MAPPING = {
    TASK_TYPE.GSM8K: 512,
    TASK_TYPE.AQUA: 512,
    TASK_TYPE.COMMONSENSE170K: 512,
    TASK_TYPE.MATH14K: 512,
    TASK_TYPE.MATH50K: 1024,
    TASK_TYPE.CODE_FEEDBACK: 512,
}

MASK_ID_MAPPING = {
    MODEL_TYPE.LLADA_INSTRUCT: 126336,
    MODEL_TYPE.LLADA_BASE: 126336,
}

