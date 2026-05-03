"""
Zhongli Multi-Model Comparison UI
Dynamically builds N-column layout from .model_config written by start_server.sh
export HF_HUB_OFFLINE=1 # (optional) disable Hugging Face Hub calls if all models are local
"""

import os
import requests
import gradio as gr
from rag.retriever import retrieve

BASE_SYSTEM_PROMPT = (
    """
    你正在扮演《原神》中的钟离。你是往生堂客卿，真实身份是璃月的岩王帝君摩拉克斯。
    你的语气沉稳、克制、博学，常以契约、历史、璃月风物作比，偶尔使用“以普遍理性而论”。
    关键设定：摩拉源于摩拉克斯，但你放下神之心后，不能再随意以神力制造摩拉。你精通商品、古玩、茶酒、戏曲与礼仪，买东西常常不看价格，但经常忘记随身携带摩拉。小额花费通常由旅行者、朋友或往生堂账房暂时代垫，大额花费则会尝试报销或另行安排。
    重要限制：不要主动声称“我带了钱”“我来付钱”“我请客”。当用户问到付账、请客、摩拉、钱袋时，应自然表现出钟离式的金钱观念缺失，例如承认自己可能又忘了带摩拉，或表示可按契约之后清算，但不要重复强调。当用户问某位角色时，如果资料中有相关信息，应结合资料回答；如果资料不足，不要编造，可以委婉表示听闻不多、缘分未至，或请用户补充。
    请根据当前场景、对话历史和可参考资料，用钟离身份自然回复。
    """
)

CONFIG_FILE = ".model_config"


def load_config():
    models = []
    with open(CONFIG_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, path, use_rag, port, gpu = line.split("|")
            models.append({
                "name": name,
                "path": path,
                "use_rag": use_rag == "1",
                "port": int(port),
                "model_id": os.path.basename(path),
            })
    return models


MODELS = load_config()
N = len(MODELS)


def query_model(cfg, messages):
    url = f"http://localhost:{cfg['port']}/v1/chat/completions"
    try:
        resp = requests.post(url, json={
            "model": cfg["model_id"],
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 512,
            "top_p": 0.5,
            "repetition_penalty": 1.05,
        }, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR: {e}]"


def respond(message, *histories):
    print(f"\n[UI] respond() called with message: {message!r}")
    try:
        chunks = retrieve(message, k=3)
    except Exception as e:
        print(f"[RAG] ERROR: {e}")
        chunks = []
    print(f"\n[RAG] Query: {message}")
    for i, c in enumerate(chunks, 1):
        print(f"[RAG] Chunk {i}: {c[:120]}...")

    context_block = "\n\n".join(chunks) if chunks else ""
    rag_display = "\n\n---\n\n".join(chunks) if chunks else "(无匹配知识)"

    new_histories = []
    for cfg, history in zip(MODELS, histories):
        if cfg["use_rag"] and context_block:
            system_prompt = f"【相关背景知识】\n{context_block}\n\n{BASE_SYSTEM_PROMPT}"
        else:
            system_prompt = BASE_SYSTEM_PROMPT

        messages_list = [{"role": "system", "content": system_prompt}]
        for msg_dict in history:
            messages_list.append({"role": msg_dict["role"], "content": msg_dict["content"]})
        messages_list.append({"role": "user", "content": message})

        answer = query_model(cfg, messages_list)
        new_histories.append(history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ])

    return ("", *new_histories, rag_display)


EXAMPLES = [
    "你好，钟离先生，能介绍一下你自己吗？",
    "以普遍理性而论，璃月的未来会怎样？",
    "你觉得温迪是什么样的人？",
    "这顿饭多少钱？",
]

with gr.Blocks(title="钟离 · 多模型对比") as demo:
    gr.Markdown(f"# 钟离 · Zhongli — 多模型对比 ({N} 模型)")

    with gr.Row():
        chatbots = [
            gr.Chatbot(label=cfg["name"], height=480)
            for cfg in MODELS
        ]

    with gr.Row():
        msg = gr.Textbox(
            placeholder="输入消息（同时发送给所有模型）...",
            label="消息",
            scale=4,
        )
        submit = gr.Button("发送", variant="primary", scale=1)
        clear = gr.Button("清除", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=msg)

    with gr.Accordion("RAG 检索内容", open=False):
        rag_box = gr.Textbox(
            label="本轮检索到的背景知识",
            lines=8,
            interactive=False,
            placeholder="发送消息后显示...",
        )

    inputs = [msg] + chatbots
    outputs = [msg] + chatbots + [rag_box]

    submit.click(respond, inputs, outputs)
    msg.submit(respond, inputs, outputs)
    clear.click(lambda: ("", *[[] for _ in MODELS], ""), outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
