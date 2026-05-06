# Zhongli AI NPC

Bring Genshin Impact's Zhongli to life using fine-tuned LLM + RAG.

**Stack:** Qwen3-8B + LoRA (LLaMA-Factory) + FAISS RAG + vLLM + Gradio

## Project Structure

```
zhongli_sft.json          # SFT training data (dialogue pairs)
zhongli_rag.json          # RAG knowledge base (character lore & opinions)
zhongli_train.yaml      # LoRA training config
setup_and_train.sh      # One-click train script
rag/
  build_index.py          # Build FAISS index from zhongli_rag.json
  retriever.py            # Query FAISS index at inference time
chat_ui.py                # Gradio chat interface
start_server.sh           # Launch vLLM + Gradio
```

## Quick Start (Remote Server with GPUs)

### 1. Train

```bash
bash setup_and_train.sh
```

This will: clone LLaMA-Factory, install deps, register dataset, run LoRA training (auto-detects free GPUs), and merge weights.

- Config: `zhongli_train.yaml` (rank=64, 5 epochs, cosine LR)
- Output: `LLaMA-Factory/output/zhongli_merged/`

### 2. Serve

```bash
bash start_server.sh
```

This will: create inference venv, install deps, build RAG index (if needed), start vLLM on port 8000, then Gradio on port 7860.

### 3. Access

```bash
ssh -L 7860:localhost:7860 user@server
# Open http://localhost:7860
```

## How RAG Works

1. `build_index.py` chunks `zhongli_rag.json` (max 400 chars), embeds with `bge-small-zh-v1.5`, saves FAISS index
2. On each user message, `retriever.py` finds top-3 relevant chunks
3. Chunks are prepended to system prompt as background knowledge

## Key Config

| Item | Value |
|------|-------|
| Base model | Qwen3-8B |
| LoRA rank / alpha | 64 / 128 |
| Embedding model | BAAI/bge-small-zh-v1.5 |
| Inference | vLLM (port 8000) |
| UI | Gradio (port 7860) |
