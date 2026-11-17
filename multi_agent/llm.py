"""Локальные LLM через llama.cpp. Одновременно загружена одна модель."""

import json
import http.client
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModelRouter:
    def __init__(self, config_path=None, work_dir=None, port=None):
        self.config = json.loads(Path(config_path or ROOT / "multi_agent/models.json").read_text())
        self.work_dir = Path(work_dir or ROOT / "multi_agent/state")
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.models = self.config["models"]
        self.roles = self.config["roles"]
        self.options = self.config["inference"]
        self.port = int(port or self.options["port"])
        judge_family = self.models[self.roles["validator"]]["family"]
        for role in ("planner", "writer"):
            if self.models[self.roles[role]]["family"] == judge_family:
                raise ValueError("Валидатор должен использовать другое семейство LLM")
        self.process = None
        self.model_key = None
        self.log_handle = None
        self.base_url = None
        # Не используем HTTP_PROXY: все запросы идут только на свой loopback-сервер.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None
        self.model_key = None

    def _start(self, model_key):
        if self.model_key == model_key and self.process and self.process.poll() is None:
            return 0.0
        self.close()
        model = self.models[model_key]
        weights = ROOT / ".cache/models" / model["filename"]
        engine = Path(os.environ.get("LLAMA_SERVER", str(ROOT / self.config["engine"]["path"])))
        if not engine.is_file() or not weights.is_file():
            raise RuntimeError("Нет движка или весов. Выполните python -m multi_agent.runtime.setup")
        if weights.stat().st_size != model["size_bytes"]:
            raise RuntimeError(f"Файл {weights.name} скачан не полностью")
        port = self.port
        try:
            with socket.socket() as probe:
                # При смене модели на том же порту остается TCP TIME_WAIT.
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Локальный port {port} занят или недоступен") from exc
        self.base_url = f"http://127.0.0.1:{port}"
        command = [
            str(engine), "--model", str(weights), "--alias", model_key,
            "--host", "127.0.0.1", "--port", str(port), "--parallel", "1",
            "--ctx-size", str(self.options["context_size"]),
            "--threads", str(self.options["threads"]),
            "--n-gpu-layers", str(self.options["gpu_layers"]),
            "--device", "none", "--no-kv-offload", "--no-op-offload", "--fit", "off",
            "--jinja", "--no-webui", "--metrics",
        ]
        self.log_handle = (self.work_dir / "llama-server.log").open("a")
        started = time.perf_counter()
        self.process = subprocess.Popen(command, stdout=self.log_handle, stderr=self.log_handle)
        deadline = time.monotonic() + self.options["timeout_s"]
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.close()
                raise RuntimeError("llama-server завершился. См. llama-server.log")
            try:
                with self.http.open(self.base_url + "/health", timeout=1) as response:
                    if response.status == 200:
                        self.model_key = model_key
                        return time.perf_counter() - started
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.15)
        self.close()
        raise RuntimeError("Истекло время загрузки локальной LLM")

    def generate(self, role, system, payload, schema):
        if role not in self.roles:
            raise ValueError(f"Неизвестная роль: {role}")
        return self.generate_model(self.roles[role], system, payload, schema)

    def generate_model(self, model_key, system, payload, schema):
        """Прямой выбор модели нужен только для одинакового benchmark трех LLM."""
        if model_key not in self.models:
            raise ValueError(f"Неизвестная модель: {model_key}")
        load_s = self._start(model_key)
        body = {
            "model": model_key,
            "messages": [
                {"role": "system", "content": system + "\nReturn JSON matching this schema: " + json.dumps(schema)},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "schema": schema},
            "temperature": self.options["temperature"],
            "seed": 42,
            "max_tokens": self.options["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
            "cache_prompt": False,
        }
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        try:
            with self.http.open(request, timeout=self.options["timeout_s"]) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read(2000).decode(errors="replace")
            raise RuntimeError(f"llama.cpp HTTP {exc.code}: {detail}") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise RuntimeError(f"Локальная LLM недоступна: {exc}") from exc
        latency_s = time.perf_counter() - started
        choice = raw["choices"][0]
        content = choice["message"].get("content") or ""
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM вернула незавершенный JSON: {content[:300]}") from exc
        if choice.get("finish_reason") == "length":
            raise RuntimeError("LLM достигла ограничения выходных токенов")
        usage = raw.get("usage", {})
        timings = raw.get("timings", {})
        model = self.models[model_key]
        return {
            "data": parsed, "model": model["name"], "model_key": model_key,
            "family": model["family"], "quantization": model["quantization"],
            "latency_s": round(load_s + latency_s, 4), "load_s": round(load_s, 4),
            "inference_s": round(latency_s, 4),
            "total_s": round(load_s + latency_s, 4),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "tokens_per_second": round(timings.get("predicted_per_second", 0), 3),
            "raw_content": content, "finish_reason": choice.get("finish_reason"),
        }
