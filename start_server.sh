#!/bin/bash
# Launch N vLLM servers + Gradio for multi-model comparison
# Greedily packs models onto fewest GPUs possible.
# Usage: bash start_server.sh
#
# Configure models in the MODELS array below:
#   "name|model_path|use_rag"
#   use_rag: 0 or 1

set -e

# ── Model configuration ────────────────────────────────────────────────────────
MODELS=(
    # "Base (Qwen3-8B)|./Qwen3-8B|0"
    # "SFT (Qwen3-8B)|./LLaMA-Factory/output/zhongli_merged_v1|0"
    # "Base+RAG (Qwen3-8B)|./Qwen3-8B|1"
    # "SFT+RAG (Qwen3-8B)|./LLaMA-Factory/output/zhongli_merged_v1|1"
    "Base (Qwen3-4B-Instruct)|./Qwen3-4B-Instruct|0"
    "SFT (Qwen3-4B-Instruct)|./LLaMA-Factory/output/zhongli_merged_v2|0"
    "Base+RAG (Qwen3-4B-Instruct)|./Qwen3-4B-Instruct|1"
    "SFT+RAG (Qwen3-4B-Instruct)|./LLaMA-Factory/output/zhongli_merged_v2|1"
    # Add more models here, e.g.:
    # "Base (Gemma)|./gemma-9b|0"
    # "SFT (Gemma)|./LLaMA-Factory/output/zhongli_gemma_merged|0"
    # "Base+RAG (Gemma)|./gemma-9b|1"
    # "SFT+RAG (Gemma)|./LLaMA-Factory/output/zhongli_gemma_merged|1"
)

# Memory required per model in MiB — used only for greedy GPU bin-packing.
# This is compared against nvidia-smi free memory, so it must account for
# GPU driver overhead (~500 MiB). Actual vLLM limit is set by gpu_memory_utilization.
# For 48GB GPU @ 0.40 utilization: 48000 * 0.40 = ~19200 MiB, minus ~500 overhead = 18700
MODEL_MIB=18700

# Starting port — models get PORT_START, PORT_START+1, PORT_START+2, ...
PORT_START=8001
# ──────────────────────────────────────────────────────────────────────────────

# Deduplicate model paths for server launching
declare -A PATH_TO_PORT
declare -a UNIQUE_PATHS
PORT_COUNTER=$PORT_START
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path use_rag <<< "$entry"
    if [[ -z "${PATH_TO_PORT[$path]}" ]]; then
        PATH_TO_PORT[$path]=$PORT_COUNTER
        UNIQUE_PATHS+=("$path")
        (( PORT_COUNTER++ ))
    fi
done
NUM_UNIQUE=${#UNIQUE_PATHS[@]}
NUM_MODELS=${#MODELS[@]}
echo ">>> Launching $NUM_MODELS model configs ($NUM_UNIQUE unique servers)"
echo ">>> GPU free memory:"
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "    GPU %s: %s / %s MiB free\n", $1, $2, $3}'

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

# Greedy GPU assignment for unique model servers only
GPU_ASSIGN_OUTPUT=$(python3 - "$NUM_UNIQUE" "$MODEL_MIB" <<'PYEOF'
import subprocess, sys

num_models = int(sys.argv[1])
model_mib  = int(sys.argv[2])

result = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
    text=True
)

free_mib = {}
for line in result.strip().splitlines():
    idx, free = line.split(", ")
    free_mib[int(idx)] = int(free)

for _ in range(num_models):
    candidates = [(f, g) for g, f in free_mib.items() if f >= model_mib]
    if not candidates:
        print("ERROR: Not enough VRAM to fit all models", file=sys.stderr)
        sys.exit(1)
    candidates.sort(reverse=True)
    chosen = candidates[0][1]
    free_mib[chosen] -= model_mib
    print(chosen)
PYEOF
)

if [ $? -ne 0 ]; then
    echo "ERROR: GPU assignment failed. Not enough free VRAM." && exit 1
fi
mapfile -t GPU_ASSIGNMENTS <<< "$GPU_ASSIGN_OUTPUT"

# Map unique paths to their assigned GPU
declare -A PATH_TO_GPU
for i in "${!UNIQUE_PATHS[@]}"; do
    PATH_TO_GPU["${UNIQUE_PATHS[$i]}"]="${GPU_ASSIGNMENTS[$i]}"
done

# Write model config file for chat_ui.py to read
CONFIG_FILE=".model_config"
> "$CONFIG_FILE"
for entry in "${MODELS[@]}"; do
    IFS='|' read -r name path use_rag <<< "$entry"
    port="${PATH_TO_PORT[$path]}"
    gpu="${PATH_TO_GPU[$path]}"
    echo "${name}|${path}|${use_rag}|${port}|${gpu}" >> "$CONFIG_FILE"
done

echo ">>> GPU assignments:"
while IFS='|' read -r name path use_rag port gpu; do
    echo "    GPU $gpu  port $port  [$name]  $path"
done < "$CONFIG_FILE"

# Start vLLM servers (one per unique model path)
VLLM_PIDS=()
declare -A LAUNCHED_PORTS
for i in "${!UNIQUE_PATHS[@]}"; do
    path="${UNIQUE_PATHS[$i]}"
    port="${PATH_TO_PORT[$path]}"
    gpu="${PATH_TO_GPU[$path]}"
    model_id=$(basename "$path")
    echo ">>> Starting vLLM [$model_id] on GPU $gpu, port $port..."
    CUDA_VISIBLE_DEVICES=$gpu python -m vllm.entrypoints.openai.api_server \
        --model "$path" \
        --host 0.0.0.0 --port "$port" \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.40 \
        --enforce-eager \
        --served-model-name "$model_id" &
    VLLM_PIDS+=($!)
    LAUNCHED_PORTS[$port]=1
    # Stagger launches to avoid simultaneous weight loading OOM
    echo ">>> Waiting 30s before next server launch..."
    sleep 30
done

# Wait for all unique servers
for port in "${!LAUNCHED_PORTS[@]}"; do
    echo ">>> Waiting for server on port $port..."
    until curl -s "http://localhost:$port/health" > /dev/null 2>&1; do
        sleep 2
    done
    echo ">>> Port $port ready!"
done

echo ">>> All models ready!"

# Print SSH tunnel command (unique ports only)
TUNNEL_ARGS="-L 7860:localhost:7860"
for path in "${UNIQUE_PATHS[@]}"; do
    port="${PATH_TO_PORT[$path]}"
    TUNNEL_ARGS="$TUNNEL_ARGS -L ${port}:localhost:${port}"
done
echo ">>> SSH tunnel: ssh $TUNNEL_ARGS user@server"

python chat_ui.py

for pid in "${VLLM_PIDS[@]}"; do
    kill "$pid" 2>/dev/null
done
