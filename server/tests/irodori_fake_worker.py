"""irodori_worker.py と同じプロトコルを話す偽のワーカー（torch なし）。テスト用。

テキストで挙動を切り替える: ``SLOW`` は 0.4 秒かかる、``ERROR`` は失敗を返す、
``CRASH`` はプロセスごと落ちる、``HANG`` は応答しない。
起動のたびに ``--log`` へ 1 行足すので、起動回数が数えられる。受け取った文も記録する。
"""

import argparse
import io
import json
import queue
import sys
import threading
import time
import wave

p = argparse.ArgumentParser()
p.add_argument("--log", required=True)
p.add_argument("--startup-delay", type=float, default=0.0)
p.add_argument("--no-ready", action="store_true")
args = p.parse_args()

with open(args.log, "a", encoding="utf-8") as f:
    f.write("start\n")
time.sleep(args.startup_delay)
if args.no_ready:
    sys.exit(3)


def send(header: dict, body: bytes = b"") -> None:
    sys.stdout.buffer.write(json.dumps(header).encode() + b"\n" + body)
    sys.stdout.buffer.flush()


def wav(seconds: float, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(seconds * sr))
    return buf.getvalue()


send({"ready": True, "sr": 24000})
requests: queue.Queue = queue.Queue()
cancelled: set[int] = set()


def reader() -> None:
    for line in sys.stdin:
        msg = json.loads(line)
        if "cancel" in msg:
            cancelled.add(msg["cancel"])
        else:
            requests.put(msg)
    requests.put(None)


threading.Thread(target=reader, daemon=True).start()
while (msg := requests.get()) is not None:
    rid, text = msg["id"], msg["text"]
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(f"text:{text}\n")
    if rid in cancelled:
        send({"id": rid, "ok": False, "error": "cancelled"})
        continue
    if "CRASH" in text:
        sys.exit(9)
    if "HANG" in text:
        time.sleep(60)
    if "SLOW" in text:
        time.sleep(0.4)
    if "ERROR" in text:
        send({"id": rid, "ok": False, "error": "boom"})
        continue
    body = wav(len(text) * 0.01)
    send({"id": rid, "ok": True, "nbytes": len(body), "sr": 24000, "synth_s": 0.0}, body)
