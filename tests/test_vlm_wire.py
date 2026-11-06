"""Wire-format tests: the real client against a local OpenAI-compatible server.

These check what MockClient cannot: the exact request shape the provider sees. A tiny
http.server stands in for the endpoint, so no network access and no API key are needed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import h5py
import numpy as np

from egowhale.annotate import Annotate
from egowhale.paths import ANNOTATION, COMPOSITE, FINAL, QUALITY, VLM_AUDIT
from egowhale.quality.vlm_audit import VlmAudit
from egowhale.vlm.cache import request_json
from egowhale.vlm.client import OpenAICompatibleClient
from egowhale.vlm.config import VLMConfig


class _Endpoint:
    """A local chat-completions endpoint that records every request and plays canned replies."""

    def __init__(self, replies: list[tuple[int, dict]]):
        self.requests: list[dict] = []
        self.replies = list(replies)
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                status, payload = (outer.replies.pop(0) if outer.replies else (200, {}))
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self) -> "_Endpoint":
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.server.shutdown()
        self.server.server_close()


def _episode(tmp_path, frames: int = 4) -> Path:
    src = tmp_path / "task" / "0.hdf5"
    src.parent.mkdir(parents=True)
    with h5py.File(src, "w") as handle:
        handle["transforms/camera"] = np.repeat(np.eye(4)[None], frames, axis=0)
    writer = cv2.VideoWriter(str(src.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return src


def _video(path: Path, frames: int = 4) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


def _config(base_url: str) -> dict:
    return {
        "vlm": {"model": "m", "base_url": base_url, "api_key_env": "TEST_WIRE_KEY", "cache": False},
        "annotate": {
            "mode": "both",
            "sample_fps": 30,
            "max_frames": 4,
            "max_subtasks": 8,
            "min_subtask_frames": 1,
            "prompt_version": "coarse_v1",
            "on_error": "keep",
        },
    }


def test_annotate_sends_an_openai_shaped_request(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_WIRE_KEY", "k")
    src = _episode(tmp_path)
    reply = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "episode_summary": "Pick up the block.",
                    "subtasks": [
                        {"start_frame": 0, "end_frame": 2, "instruction": "reach"},
                        {"start_frame": 2, "end_frame": 4, "instruction": "lift"},
                    ],
                })
            }
        }]
    }
    with _Endpoint([(200, reply)]) as endpoint:
        Annotate(_config(endpoint.url)).run(src, tmp_path / "out")
    report = json.loads((tmp_path / "out" / ANNOTATION).read_text())
    assert report["status"] == "ok" and report["episode_summary"] == "Pick up the block."

    assert len(endpoint.requests) == 1
    request = endpoint.requests[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["headers"].get("Authorization") == "Bearer k"
    assert request["headers"].get("Content-Type") == "application/json"
    body = request["body"]
    assert body["model"] == "m"
    message = body["messages"][0]
    assert message["role"] == "user"
    content = message["content"]
    assert [item["type"] for item in content] == ["image_url", "image_url", "image_url", "image_url", "text"]
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in content[:-1])
    assert "Sampled frames" in content[-1]["text"]


def test_vlm_audit_runs_end_to_end_over_http(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_WIRE_KEY", "k")
    out = tmp_path / "out"
    out.mkdir()
    quality = {
        "keep_l1_l2": True,
        "keep": True,
        "frames": 4,
        "fps": 30.0,
        "approach": 0,
        "description": "Pick up the block.",
        "description_source": "hdf5",
    }
    (out / QUALITY).parent.mkdir(parents=True, exist_ok=True)
    (out / QUALITY).write_text(json.dumps(quality))
    _video(out / COMPOSITE)
    config = {
        "vlm": {"model": "m", "base_url": "", "api_key_env": "TEST_WIRE_KEY", "cache": False},
        "audit": {"enabled": True, "sample_fps": 1, "max_frames": 4, "prompt_version": "audit_v1", "fail_open": True},
    }
    reply = {"choices": [{"message": {"content": json.dumps({"is_consistent": True, "confidence": 0.9})}}]}
    with _Endpoint([(200, reply)]) as endpoint:
        config["vlm"]["base_url"] = endpoint.url
        VlmAudit(config).run(tmp_path / "0.hdf5", out)
    audit = json.loads((out / VLM_AUDIT).read_text())
    final = json.loads((out / FINAL).read_text())
    assert audit["status"] == "ok" and audit["is_consistent"] is True
    assert final["decision"] == "keep"
    prompt = endpoint.requests[0]["body"]["messages"][0]["content"][-1]["text"]
    assert "Task Description: Pick up the block." in prompt


def test_429_is_retried_over_real_http():
    def _text(value: str) -> dict:
        return {"choices": [{"message": {"content": value}}]}

    with _Endpoint([(429, {"error": {"message": "slow down"}}), (200, _text("hello"))]) as endpoint:
        client = OpenAICompatibleClient(
            VLMConfig(base_url=endpoint.url, model="m", api_key="k", retries=1, backoff_s=0)
        )
        assert client.complete([{"role": "user", "content": "hi"}]) == "hello"
    assert len(endpoint.requests) == 2


def test_repair_round_goes_back_over_the_wire():
    def _text(value: str) -> dict:
        return {"choices": [{"message": {"content": value}}]}

    with _Endpoint([(200, _text("sure, here you go")), (200, _text('{"answer": 1}'))]) as endpoint:
        client = OpenAICompatibleClient(
            VLMConfig(base_url=endpoint.url, model="m", api_key="k", retries=0)
        )
        payload = request_json(client, [{"role": "user", "content": "hi"}], None, schema='{"answer": 0}')
    assert payload == {"answer": 1}
    assert len(endpoint.requests) == 2
    follow_up = endpoint.requests[1]["body"]["messages"][-1]
    assert follow_up["role"] == "user"
    assert "not valid JSON" in follow_up["content"]
