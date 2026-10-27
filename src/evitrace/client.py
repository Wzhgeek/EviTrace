"""Explicitly enabled API execution with payload caching and budget reservations."""
from pathlib import Path
import json
import math
import os
import sqlite3
import threading
import urllib.error
import urllib.request
from .io import canonical, sha


class BudgetExceeded(RuntimeError):
    pass


class APIClient:
    def __init__(self, cache, budget_cny, config):
        if not math.isfinite(budget_cny) or budget_cny <= 0:
            raise ValueError("A positive --budget-cny is required for network execution")
        self.path = Path(cache)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = budget_cny
        self.config = config
        self.lock = threading.Lock()
        with self.database() as db:
            db.execute("CREATE TABLE IF NOT EXISTS requests (key TEXT PRIMARY KEY, status TEXT, reserved REAL, cost REAL, response TEXT)")

    def database(self):
        return sqlite3.connect(self.path, timeout=60)

    def request(self, endpoint, payload, key_env, reserve, charge):
        key = os.environ.get(key_env)
        if not key:
            raise ValueError(f"Environment variable {key_env} is required")
        body = canonical(payload)
        signature = sha(endpoint.encode() + b"\0" + body)
        with self.lock, self.database() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute("SELECT status,response FROM requests WHERE key=?", (signature,)).fetchone()
            if previous:
                if previous[0] != "complete":
                    raise RuntimeError("An incomplete identical request needs manual inspection before resubmission")
                return json.loads(previous[1])
            spent = db.execute("SELECT COALESCE(SUM(CASE WHEN status='complete' THEN cost ELSE reserved END),0) FROM requests").fetchone()[0]
            if spent + reserve > self.limit:
                raise BudgetExceeded("The configured CNY limit prevents further requests")
            db.execute("INSERT INTO requests VALUES (?,?,?,?,?)", (signature, "pending", reserve, 0., None))
        request = urllib.request.Request(endpoint, body, {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.config["timeout_seconds"]) as reply:
                response = json.load(reply)
            cost = charge(response)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("Invalid API usage charge")
            with self.lock, self.database() as db:
                db.execute("UPDATE requests SET status='complete',cost=?,response=? WHERE key=?", (cost, json.dumps(response), signature))
            return response
        except Exception as exc:
            with self.lock, self.database() as db:
                db.execute("UPDATE requests SET status='uncertain' WHERE key=?", (signature,))
            # Preserve the reservation and expose no response body or credentials.
            raise RuntimeError(f"Request failed ({type(exc).__name__}); inspect the local cache before retrying") from None

    def chat(self, messages, max_tokens):
        cfg = self.config
        payload = {"model": cfg["model"], "messages": messages, "temperature": 0,
                   "thinking": {"type": "disabled"}, "stream": False, "max_tokens": max_tokens}
        pricing = cfg["pricing_cny_per_million"]
        reserve = ((len(canonical(payload)) + 256) * pricing["input_miss"] + max_tokens * pricing["output"]) / 1e6
        def charge(response):
            usage = response.get("usage", {})
            if not isinstance(usage.get("prompt_tokens"), int) or not isinstance(usage.get("completion_tokens"), int):
                return reserve
            return (usage["prompt_tokens"] * pricing["input_miss"] + usage["completion_tokens"] * pricing["output"]) / 1e6
        response = self.request(cfg["base_url"].rstrip("/") + "/chat/completions", payload, "DEEPSEEK_API_KEY", reserve, charge)
        choice = response["choices"][0]
        if response.get("model") != cfg["model"] or choice.get("finish_reason") != "stop":
            raise ValueError("Response model or completion status does not match the requested contract")
        content = choice["message"].get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Empty model response")
        return content.strip(), response.get("usage", {})

    def embed(self, text):
        cfg = self.config["embedding"]
        payload = {"model": "text-embedding-v4", "input": text, "dimensions": 1024, "encoding_format": "float"}
        price = cfg["price_cny_per_million"]
        reserve = (len(text.encode("utf-8")) + 256) * price / 1e6
        def charge(response):
            tokens = response.get("usage", {}).get("total_tokens")
            return tokens * price / 1e6 if isinstance(tokens, int) and tokens >= 0 else reserve
        response = self.request(cfg["base_url"].rstrip("/") + "/embeddings", payload, "DASHSCOPE_API_KEY", reserve, charge)
        vector = response["data"][0]["embedding"]
        if len(vector) != 1024 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector):
            raise ValueError("Invalid embedding vector")
        return vector
