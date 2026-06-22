#!/usr/bin/env python3
"""
serve.py — Fugu as a single OpenAI-compatible model endpoint.

This is Fugu's real product surface: "one model to command them all". A client
POSTs to /v1/chat/completions as if calling one model; internally the TRINITY
coordinator (Qwen3-0.6B + model_iter_60.npy) routes each turn to a worker from a
real pool (via litellm) and runs the step_trinity loop until a verifier accepts.
The caller never sees the pool.

stdlib http.server only — no FastAPI/uvicorn (ponytail: a router endpoint needs
a socket and a JSON handler, not a web framework).

Run:
  FUGU_API_KEY=... FUGU_BASE_URL=... \
  python serve.py --model <qwen3-0.6b dir> --vector model_iter_60.npy \
                  --slot-models <csv of litellm worker ids> --port 8088

Query:
  curl localhost:8088/v1/chat/completions -d '{"messages":[{"role":"user","content":"..."}]}'
"""
from __future__ import annotations
import argparse, json, os, sys, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# reuse the faithful implementation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini import FuguRouter, Coordinator, LiteLLMWorker, MockWorker, DEFAULT_SLOT_LABELS

ROUTER: FuguRouter | None = None
WORKER = None
MODEL_NAME = "fugu"
MAX_TURNS = 5


def _chat_response(text: str, model: str, usage_turns: int) -> dict:
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:24],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        # surface the orchestration depth without exposing which workers ran
        "usage": {"fugu_turns": usage_turns},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "openfugu"}]})
        elif self.path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": MODEL_NAME})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": "not found"}); return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            messages = req.get("messages", [])
            if not messages:
                self._send(400, {"error": "messages required"}); return
            # the user query = last user message; coordinator runs the full loop
            query = next((m["content"] for m in reversed(messages)
                          if m.get("role") == "user"), "")
            coord = Coordinator(ROUTER, WORKER, max_turns=MAX_TURNS, sample=True)
            res = coord.run(query, verbose=False)
            self._send(200, _chat_response(res.final, req.get("model", MODEL_NAME),
                                           len(res.turns)))
        except Exception as e:
            self._send(500, {"error": str(e)})

    def log_message(self, *a):       # quiet
        pass


def main():
    global ROUTER, WORKER, MAX_TURNS
    ap = argparse.ArgumentParser(description="Serve Fugu as one OpenAI-compatible model.")
    ap.add_argument("--model", required=True, help="Qwen3-0.6B dir")
    ap.add_argument("--vector", default="model_iter_60.npy")
    ap.add_argument("--slot-models", metavar="CSV", help="litellm worker ids; omit for mock")
    ap.add_argument("--port", type=int, default=8088)
    ap.add_argument("--max-turns", type=int, default=5)
    args = ap.parse_args()
    MAX_TURNS = args.max_turns

    print(f"[serve] loading TRINITY router ({args.model}) ...", flush=True)
    ROUTER = FuguRouter(args.model, args.vector, seed=0)
    if args.slot_models:
        WORKER = LiteLLMWorker(slot_models=args.slot_models.split(","))
        print(f"[serve] worker pool: litellm ({len(args.slot_models.split(','))} slots)", flush=True)
    else:
        WORKER = MockWorker()
        print("[serve] worker pool: MOCK (no --slot-models given)", flush=True)

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[serve] Fugu listening on :{args.port} — POST /v1/chat/completions", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
