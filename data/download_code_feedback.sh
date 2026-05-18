#!/bin/bash
# Download code_feedback dataset from Hugging Face
# Usage: HF_TOKEN=your_token bash data/download_code_feedback.sh
#    or: export HF_TOKEN=your_token && bash data/download_code_feedback.sh
#
# Get your token at https://huggingface.co/settings/tokens

export HF_ENDPOINT=https://hf-mirror.com   # optional: use mirror for faster download in China

if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN is not set."
    echo "Usage: HF_TOKEN=<your_token> bash data/download_code_feedback.sh"
    echo "Get your token at: https://huggingface.co/settings/tokens"
    exit 1
fi

huggingface-cli download m-a-p/CodeFeedback-Filtered-Instruction \
    --repo-type dataset \
    --token "$HF_TOKEN" \
    --local-dir data/llm_adapt/code_feedback

echo "Done. Dataset saved to data/llm_adapt/code_feedback/"
