# サーバーのデプロイ（systemd）

研究室の GPU マシンで `reachy_server` を常駐させる。GPU は共用機で、Irodori-TTS は専用の Python 環境（`~/Irodori-TTS/.venv`）を使うので、Docker ではなく **systemd のユーザーユニット**にしている（root 不要）。

## 初回の手順
1. 依存を入れる: `uv sync --package reachy-server --group gpu`
2. `~/Irodori-TTS` に専用の環境を作る（`uv sync --extra cu128`）。話者の参照音声を `data/server/voice/reference.wav` に置く（Git には入れない。実在の人の声は使わない）
3. `.env` を作る（`.env.example` を参照）。最低限:
   ```
   REACHY_SERVER_TOKEN=<16 文字以上。ロボットと同じ値>
   OPENAI_API_KEY=<サーバーだけに置く>
   REACHY_SERVER_STT=faster-whisper
   REACHY_SERVER_TTS=irodori
   REACHY_SERVER_LLM=openai
   ```
   `.env` は `chmod 600`。
4. モデルを事前に取得する: `uv run --package reachy-server --group gpu python scripts/deploy/prefetch_models.py`
5. ユニットを置いて起動する:
   ```bash
   mkdir -p ~/.config/systemd/user
   cp deploy/server/reachy-server.service ~/.config/systemd/user/
   systemctl --user daemon-reload
   systemctl --user enable --now reachy-server
   sudo loginctl enable-linger "$USER"   # ログインしていなくても、再起動後に自動で起動する
   ```
6. 確認: `curl -s localhost:8765/healthz`、ログは `journalctl --user -u reachy-server -f`

## 運用
- 更新: `git pull && uv sync --package reachy-server --group gpu && systemctl --user restart reachy-server`
- 設定を変えたら `systemctl --user restart reachy-server`
- 落ちたら 10 秒後に自動で起こし直す（5 分に 5 回続けて落ちたら止まる。`systemctl --user reset-failed reachy-server` で戻す）
- GPU は起動のたびに空いているものを選ぶ（`REACHY_SERVER_GPU_MIN_FREE_MIB`、手で選ぶなら `CUDA_VISIBLE_DEVICES`）。空きがなければ起動に失敗し、10 秒おきにやり直す
- ロボットからは `ws://<このマシン>:8765/v1/session` に繋ぐ。ポートを LAN に開けるときは、トークンを必ず設定する
