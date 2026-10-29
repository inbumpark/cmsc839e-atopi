"""Thin wrapper around the Ollama chat API with image encoding and JSON parsing."""
import base64
import io
import json
import re
import time
from functools import lru_cache

from PIL import Image


@lru_cache(maxsize=256)
def encode_image(path, max_side):
    img = Image.open(path)
    img = img.convert("RGB")
    if max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def parse_json(text):
    """Parse a JSON object from model output, tolerating code fences or surrounding prose."""
    text = (text or "").strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


class OllamaClient:
    def __init__(self, cfg):
        import ollama
        self._ollama = ollama
        self.client = ollama.Client(host=cfg["ollama_host"])
        self.model = cfg["model"]
        self.think = cfg.get("think", False)
        self.max_side = cfg.get("image_max_side")
        self.base_options = {"temperature": cfg["temperature"], "num_ctx": cfg.get("num_ctx", 8192)}
        self.keep_alive = cfg.get("keep_alive", "30m")

    def images(self, paths):
        return [encode_image(p, self.max_side) for p in paths]

    def chat(self, messages, schema, seed):
        """Send one chat turn. Returns (content, latency_seconds)."""
        kwargs = dict(model=self.model, messages=messages, format=schema,
                      options={**self.base_options, "seed": seed}, keep_alive=self.keep_alive)
        if self.think is not None:
            kwargs["think"] = self.think
        t0 = time.time()
        try:
            resp = self.client.chat(**kwargs)
        except self._ollama.ResponseError as e:
            # Models without a thinking toggle reject the `think` argument; retry without it.
            if "think" in kwargs and "think" in str(e).lower():
                self.think = None
                kwargs.pop("think")
                resp = self.client.chat(**kwargs)
            else:
                raise
        return resp.message.content, time.time() - t0

    def check(self):
        """Fail fast if the server is down or the model is not pulled."""
        try:
            names = {m.model for m in self.client.list().models}
        except Exception as e:
            raise SystemExit(f"Cannot reach Ollama at {self.client._client.base_url}: {e}\n"
                             "Start it with `brew services start ollama` (or `ollama serve`).")
        if self.model not in names and f"{self.model}:latest" not in names:
            raise SystemExit(f"Model '{self.model}' is not pulled. Run `ollama pull {self.model}`.")
