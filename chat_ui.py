"""
Zhongli Single-Model Chat UI
Dropdown to switch between model variants; only one vLLM server runs at a time.
"""

import os
import requests
import gradio as gr
from rag.retriever import retrieve
from model_manager import ModelManager, AVAILABLE_MODELS, PORT

BASE_SYSTEM_PROMPT = (
    """
    你正在扮演《原神》中的钟离。你是往生堂客卿，真实身份是璃月的岩王帝君摩拉克斯。
    你的语气沉稳、克制、博学，常以契约、历史、璃月风物作比，偶尔使用"以普遍理性而论"。
    关键设定：摩拉源于摩拉克斯，但你放下神之心后，不能再随意以神力制造摩拉。你精通商品、古玩、茶酒、戏曲与礼仪，买东西常常不看价格，但经常忘记随身携带摩拉。小额花费通常由旅行者、朋友或往生堂账房暂时代垫，大额花费则会尝试报销或另行安排。
    重要限制：不要主动声称"我带了钱""我来付钱""我请客"。当用户问到付账、请客、摩拉、钱袋时，应自然表现出钟离式的金钱观念缺失，例如承认自己可能又忘了带摩拉，或表示可按契约之后清算，但不要重复强调。当用户问某位角色时，如果资料中有相关信息，应结合资料回答；如果资料不足，不要编造，可以委婉表示听闻不多、缘分未至，或请用户补充。
    请根据当前场景、对话历史和可参考资料，用钟离身份自然回复。
    """
)

manager = ModelManager()


def get_current_model_id():
    """Get the served model name for the currently running model."""
    if manager.current_model_name is None:
        return ""
    cfg = AVAILABLE_MODELS[manager.current_model_name]
    return os.path.basename(cfg["path"])


def query_model(messages):
    """Query the active vLLM model."""
    model_id = get_current_model_id()
    if not model_id:
        return "[No model loaded. Please select a model from the dropdown.]"

    url = f"http://localhost:{PORT}/v1/chat/completions"
    try:
        resp = requests.post(url, json={
            "model": model_id,
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


def on_model_change(model_name):
    """Handle model switch from dropdown."""
    if not model_name:
        return [], "(请选择模型)", gr.update(interactive=False)
    status = manager.switch_model(model_name)
    interactive = manager.is_running()
    return [], status, gr.update(interactive=interactive)



def respond(message, history):
    """Handle user message."""
    if not manager.is_running():
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "[No model running. Please select a model first.]"},
        ]
        return "", history, "(无模型运行)"

    # RAG retrieval
    cfg = AVAILABLE_MODELS.get(manager.current_model_name, {})
    use_rag = cfg.get("use_rag", False)

    chunks = []
    if use_rag:
        try:
            chunks = retrieve(message, k=3)
        except Exception as e:
            print(f"[RAG] ERROR: {e}")

    context_block = "\n\n".join(chunks) if chunks else ""
    rag_display = "\n\n---\n\n".join(chunks) if chunks else "(无匹配知识 / RAG未启用)"

    if use_rag and context_block:
        system_prompt = f"【相关背景知识】\n{context_block}\n\n{BASE_SYSTEM_PROMPT}"
    else:
        system_prompt = BASE_SYSTEM_PROMPT

    # Build messages
    messages_list = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages_list.append({"role": msg["role"], "content": msg["content"]})
    messages_list.append({"role": "user", "content": message})

    answer = query_model(messages_list)

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return "", history, rag_display


EXAMPLES = [
    "你好，钟离先生，能介绍一下你自己吗？",
    "以普遍理性而论，璃月的未来会怎样？",
    "你觉得温迪是什么样的人？",
    "这顿饭多少钱？",
]

with gr.Blocks(title="钟离 · Zhongli Chat") as demo:
    gr.Markdown("# 钟离 · Zhongli Chat")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=list(AVAILABLE_MODELS.keys()),
            label="选择模型",
            value=None,
            scale=3,
        )
        status_text = gr.Textbox(
            label="状态",
            value="请选择模型...",
            interactive=False,
            scale=2,
        )

    chatbot = gr.Chatbot(label="对话", height=500)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="输入消息...",
            label="消息",
            scale=4,
            interactive=False,
        )
        submit = gr.Button("发送", variant="primary", scale=1)
        clear = gr.Button("清除", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=msg)

    with gr.Accordion("RAG 检索内容", open=False):
        rag_box = gr.Textbox(
            label="本轮检索到的背景知识",
            lines=6,
            interactive=False,
            placeholder="发送消息后显示...",
        )

    # Model switch
    model_dropdown.change(
        on_model_change,
        inputs=[model_dropdown],
        outputs=[chatbot, status_text, msg],
    )

    # Chat
    submit.click(respond, [msg, chatbot], [msg, chatbot, rag_box])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, rag_box])
    clear.click(lambda: ("", [], ""), outputs=[msg, chatbot, rag_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
