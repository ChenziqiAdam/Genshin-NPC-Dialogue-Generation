#!/bin/bash
# Launch Zhongli Chat UI (single-model mode)
# The UI manages vLLM servers — only one model runs at a time.
# Usage: bash start_server.sh

set -e
export HF_HUB_OFFLINE=1 

# Create inference venv
VENV_DIR=".venv_inference"
if [ ! -d "$VENV_DIR" ]; then
    echo ">>> Creating inference venv..."
    uv venv "$VENV_DIR" --python 3.11
fi
source "$VENV_DIR/bin/activate"

if ! python -c "import vllm, gradio, openai, sentence_transformers, faiss" 2>/dev/null; then
    echo ">>> Installing inference dependencies..."
    uv pip install vllm gradio openai sentence-transformers faiss-cpu
else
    echo ">>> Dependencies already installed, skipping."
fi

# Build RAG index if needed
if [ ! -f "rag/zhongli.index" ]; then
    echo ">>> Building RAG index..."
    python rag/build_index.py
else
    echo ">>> RAG index already exists, skipping."
fi

echo ">>> Starting Zhongli Chat UI (single-model mode)..."
echo ">>> Select a model from the dropdown to load it."
echo ">>> SSH tunnel: ssh -L 7860:localhost:7860 user@server"

python chat_ui.py
