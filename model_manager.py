"""
Model Manager — launches and switches vLLM servers one at a time.
"""

import os
import subprocess
import time
import signal
import requests


AVAILABLE_MODELS = {
    "Base (Qwen3-8B)": {"path": "./Qwen3-8B", "use_rag": False},
    "SFT (Qwen3-8B)": {"path": "./LLaMA-Factory/output/zhongli_merged_v1", "use_rag": False},
    "Base+RAG (Qwen3-8B)": {"path": "./Qwen3-8B", "use_rag": True},
    "SFT+RAG (Qwen3-8B)": {"path": "./LLaMA-Factory/output/zhongli_merged_v1", "use_rag": True},
    "Base (Qwen3-4B-Instruct)": {"path": "./Qwen3-4B-Instruct", "use_rag": False},
    "SFT (Qwen3-4B-Instruct)": {"path": "./LLaMA-Factory/output/zhongli_merged_v2", "use_rag": False},
    "Base+RAG (Qwen3-4B-Instruct)": {"path": "./Qwen3-4B-Instruct", "use_rag": True},
    "SFT+RAG (Qwen3-4B-Instruct)": {"path": "./LLaMA-Factory/output/zhongli_merged_v2", "use_rag": True},
}

PORT = 8001


def _pick_gpu():
    """Pick the GPU with the most free VRAM."""
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        best_gpu, best_free = 0, 0
        for line in result.strip().splitlines():
            idx, free = line.split(", ")
            if int(free) > best_free:
                best_gpu, best_free = int(idx), int(free)
        return best_gpu
    except Exception:
        return 0


class ModelManager:
    def __init__(self):
        self.process = None
        self.current_model_name = None
        self.gpu_id = _pick_gpu()

    def switch_model(self, model_name: str) -> str:
        """Switch to a different model. Returns status message."""
        if model_name == self.current_model_name:
            return f"Already running: {model_name}"

        cfg = AVAILABLE_MODELS.get(model_name)
        if cfg is None:
            return f"Unknown model: {model_name}"

        # If only the RAG flag changed (same model path), skip restart
        if self.current_model_name is not None and self.is_running():
            current_cfg = AVAILABLE_MODELS.get(self.current_model_name, {})
            if current_cfg.get("path") == cfg["path"]:
                self.current_model_name = model_name
                rag_status = "RAG enabled" if cfg["use_rag"] else "RAG disabled"
                return f"Switched to: {model_name} ({rag_status}, no restart needed)"

        self.stop()

        model_path = cfg["path"]
        model_id = os.path.basename(model_path)

        print(f"[ModelManager] Launching {model_name} (path={model_path}, GPU={self.gpu_id})")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)

        self.process = subprocess.Popen(
            [
                "python", "-m", "vllm.entrypoints.openai.api_server",
                "--model", model_path,
                "--host", "0.0.0.0",
                "--port", str(PORT),
                "--tensor-parallel-size", "1",
                "--max-model-len", "4096",
                "--gpu-memory-utilization", "0.85",
                "--enforce-eager",
                "--served-model-name", model_id,
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            self._wait_healthy(timeout=180)
        except TimeoutError:
            self.stop()
            return f"Failed to start {model_name} (timeout)"

        self.current_model_name = model_name
        print(f"[ModelManager] {model_name} is ready on port {PORT}")
        return f"Switched to: {model_name}"

    def stop(self):
        """Terminate the current vLLM process."""
        if self.process is not None:
            print("[ModelManager] Stopping current model...")
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None
            self.current_model_name = None
            # Give the GPU a moment to release memory
            time.sleep(3)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def _wait_healthy(self, timeout=180):
        """Wait until the vLLM server responds to /health."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                r = requests.get(f"http://localhost:{PORT}/health", timeout=2)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"Server did not become healthy within {timeout}s")
