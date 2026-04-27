#!/bin/bash
# Launch vLLM + Gradio on the remote server
# Usage: bash start_server.sh
#
# This creates a SEPARATE venv for inference to avoid
# conflicting with LLaMA-Factory's transformers version.

set -e

MODEL_PATH="./LLaMA-Factory/output/zhongli_merged"

# Auto-detect a free GPU
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', ' '$2 < 1000 {print $1; exit}')

if [ -z "$FREE_GPU" ]; then
    echo "ERROR: No free GPU found!" && exit 1
fi
echo ">>> Using GPU $FREE_GPU"

# Create a separate inference venv (won't conflict with LLaMA-Factory)
VENV_DIR=".venv_inference"
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating inference venv..."
    uv venv "$VENV_DIR" --python 3.11
fi
source "$VENV_DIR/bin/activate"

# Install inference deps only if not already installed
if ! python -c "import vllm, gradio, openai" 2>/dev/null; then
    echo ">>> Installing inference dependencies..."
    uv pip install vllm gradio openai
else
    echo ">>> Dependencies already installed, skipping."
fi

# Start vLLM in background
echo ">>> Starting vLLM server on port 8000..."
CUDA_VISIBLE_DEVICES=$FREE_GPU python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 &
VLLM_PID=$!

# Wait for vLLM to be ready
echo ">>> Waiting for vLLM to load model..."
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
    sleep 2
done
echo ">>> vLLM ready!"

# Start Gradio
echo ">>> Starting Gradio UI on port 7860..."
echo ">>> Access via: ssh -L 7860:localhost:7860 -L 8000:localhost:8000 user@server"
python chat_ui.py

# Cleanup
kill $VLLM_PID 2>/dev/null
