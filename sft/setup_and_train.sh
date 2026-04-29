#!/bin/bash
# Zhongli LoRA Training Setup & Launch Script
# Run this on the remote server with 5x A800 GPUs

set -e

# ============ Step 1: Install LLaMA-Factory ============
if [ ! -d "LLaMA-Factory" ]; then
    echo ">>> Cloning LLaMA-Factory..."
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi

cd LLaMA-Factory
echo ">>> Installing LLaMA-Factory..."
pip install -e ".[torch,metrics,deepspeed]"

# ============ Step 2: Download Qwen3-8B ============
echo ">>> Downloading Qwen3-8B (skip if already cached)..."
hf download Qwen/Qwen3-8B

# ============ Step 3: Prepare dataset ============
echo ">>> Copying dataset and config..."
cp ../zhongli_sft.json data/
# Merge our dataset entry into LLaMA-Factory's dataset_info.json
python3 -c "
import json
with open('data/dataset_info.json', 'r') as f:
    info = json.load(f)
with open('../dataset_info.json', 'r') as f:
    new_entry = json.load(f)
info.update(new_entry)
with open('data/dataset_info.json', 'w') as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
print('Dataset registered successfully.')
"

# ============ Step 4: Launch training ============
cp ../zhongli_train.yaml .

# Auto-detect free GPUs (memory usage < 1000 MiB)
FREE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 {print $1}' | paste -sd,)
NUM_FREE=$(echo "$FREE_GPUS" | tr ',' '\n' | wc -l | tr -d ' ')

if [ -z "$FREE_GPUS" ]; then
    echo "ERROR: No free GPUs found!" && exit 1
fi

echo ">>> Found ${NUM_FREE} free GPU(s): ${FREE_GPUS}"
echo ">>> Starting LoRA training..."
CUDA_VISIBLE_DEVICES=$FREE_GPUS FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=$NUM_FREE \
    llamafactory-cli train zhongli_train.yaml

echo ">>> Training complete! LoRA adapter saved to ./output/zhongli_lora"

# ============ Step 5: Merge LoRA weights ============
echo ">>> Merging LoRA weights..."
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen3-8B \
    --adapter_name_or_path ./output/zhongli_lora \
    --template qwen \
    --export_dir ./output/zhongli_merged

echo ">>> Done! Merged model saved to ./output/zhongli_merged"
