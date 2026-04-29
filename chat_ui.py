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
from rag.retriever import retrieve

BASE_SYSTEM_PROMPT = (
    "你正在扮演《原神》中的钟离。你是往生堂客卿，真实身份是璃月的岩王帝君（摩拉克斯）。"
    "你性格沉稳、学识渊博，说话常带古风，缺乏世俗的金钱观念，经常使用'以普遍理性而论'等词汇。"
    "请根据当前的场景和对话历史，用钟离的语气进行回复。"
)

VLLM_URL = "http://localhost:8000/v1/chat/completions"


def chat(message, history):
    # Retrieve relevant chunks
    chunks = retrieve(message, k=3)

    # Log to console
    print(f"\n[RAG] Query: {message}")
    for i, chunk in enumerate(chunks, 1):
        print(f"[RAG] Chunk {i}: {chunk[:120]}...")

    # Build system prompt with context
    if chunks:
        context = "\n\n".join(chunks)
        system_prompt = f"【相关背景知识】\n{context}\n\n{BASE_SYSTEM_PROMPT}"
    else:
        system_prompt = BASE_SYSTEM_PROMPT

    # Build message list
    messages = [{"role": "system", "content": system_prompt}]
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
    answer = resp.json()["choices"][0]["message"]["content"]

    # Format RAG display for frontend
    if chunks:
        rag_display = "\n\n---\n\n".join(chunks)
    else:
        rag_display = "(无匹配知识)"

    return answer, rag_display


EXAMPLES = [
    "你好，钟离先生，能介绍一下你自己吗？",
    "以普遍理性而论，璃月的未来会怎样？",
    "你觉得温迪是什么样的人？",
    "这顿饭多少钱？",
]

with gr.Blocks(title="钟离 · Zhongli") as demo:
    gr.Markdown("# 钟离 · Zhongli\n与《原神》钟离对话 — 基于 Qwen3-8B LoRA 微调模型")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="输入消息...", label="你的消息", lines=1)
            with gr.Row():
                submit = gr.Button("发送", variant="primary")
                clear = gr.Button("清除")

            gr.Examples(examples=EXAMPLES, inputs=msg)

        with gr.Column(scale=1):
            with gr.Accordion("RAG 检索内容", open=True):
                rag_box = gr.Textbox(
                    label="本轮检索到的背景知识",
                    lines=20,
                    interactive=False,
                    placeholder="发送消息后显示检索到的知识片段...",
                )

    def respond(message, history):
        answer, rag_display = chat(message, history)
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return "", history, rag_display

    submit.click(respond, [msg, chatbot], [msg, chatbot, rag_box])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, rag_box])
    clear.click(lambda: ([], ""), outputs=[chatbot, rag_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
