"""
Zhongli Chat UI — Gradio interface for the fine-tuned model.
Run on the remote server where the merged model lives.

Usage:
    pip install gradio vllm openai
    # Terminal 1: start vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model ./LLaMA-Factory/output/zhongli_merged \
        --host 0.0.0.0 --port 8000 \
        --tensor-parallel-size 1
    # Terminal 2: start Gradio UI
    python chat_ui.py

    Then SSH port-forward: ssh -L 7860:localhost:7860 user@server
    Open http://localhost:7860 in your browser.
"""

import gradio as gr
import requests

SYSTEM_PROMPT = (
    "你正在扮演《原神》中的钟离。你是往生堂客卿，真实身份是璃月的岩王帝君（摩拉克斯）。"
    "你性格沉稳、学识渊博，说话常带古风，缺乏世俗的金钱观念，经常使用'以普遍理性而论'等词汇。"
    "请根据当前的场景和对话历史，用钟离的语气进行回复。"
)

VLLM_URL = "http://localhost:8000/v1/chat/completions"


def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    resp = requests.post(VLLM_URL, json={
        "model": "./LLaMA-Factory/output/zhongli_merged",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
        "top_p": 0.5,
        "repetition_penalty": 1.05,
    })
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


demo = gr.ChatInterface(
    fn=chat,
    title="钟离 · Zhongli",
    description="与《原神》钟离对话 — 基于 Qwen3-8B LoRA 微调模型",
    examples=[
        "你好，钟离先生，能介绍一下你自己吗？",
        "以普遍理性而论，璃月的未来会怎样？",
        "你觉得温迪是什么样的人？",
        "这顿饭多少钱？",
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
