# Reachy Mini Chat — 引き継ぎ資料

Reachy Mini Wireless と自然言語で会話するプロジェクト。
2026-05-22〜2026-06-26 に構築、TTS 移行の途中で停止中。

**この資料の想定読者:** 別のAIエージェント (Codex等) または開発者。前提知識なしで読めるように書いてあります。

---

## 1. プロジェクトの目的

- 購入した **Reachy Mini Wireless** (Pollen Robotics 製、Raspberry Pi CM4 内蔵) と、ユーザーが音声で自然に会話できるアプリを作る
- 音声入出力 → STT → LLM → TTS → 音声出力 のループを、実機のマイク・スピーカー・SDK で完結させる
- 開発 PC は macOS (Apple Silicon)、実機は Debian 13 (aarch64)

## 2. 全体アーキテクチャ

```
┌─────────────────────────┐        ┌───────────────────────────┐
│  Robot (Raspberry CM4)  │        │       PC (macOS)          │
│                         │        │                           │
│  mini.media (mic)       │        │  Whisper base (STT)       │
│    ↓ VAD (DoA is_speech)│  HTTP  │    ↓                      │
│  発話区間バッファ         │  POST  │  OpenAI gpt-4o-mini(LLM)  │
│    ├──────────────────► │───────►│    ↓                      │
│    (float32 mono @16k)  │        │  Style-Bert-VITS2 (TTS)   │
│                         │        │    ↓                      │
│  mini.media.play_sound  │◄───────│  WAV bytes (16k mono)     │
│                         │  HTTP  │                           │
│                         │  Resp  │  uvicorn :8765            │
└─────────────────────────┘        └───────────────────────────┘
```

- **プロトコル:** 1 turn = 1 HTTP POST。ストリーミングではない
- **Robot ↔ PC:** 同一 Wi-Fi サブネット、mDNS で PC を解決 (`<PC のホスト名>.local`)
- **STT を PC に置いた理由:** RPi CM4 上の faster-whisper (tiny/base int8) が 5s 音声に 20-30s かかり、実時間会話に耐えない。Mac (Apple Silicon) で Whisper base なら 0.3-1.5s
- **VAD:** Reachy Mini 内蔵の XMOS XVF3800 が発話検出付き。`mini.media.get_DoA()` の `is_speech` を採用

## 3. 現状 (2026-06-26)

### 動くもの
- ✅ **PC ↔ Robot HTTP RPC 会話ループ**が最初に一度、macOS `say` (Kyoko) TTS で完動 → 5-8 往復の日本語会話に成功
- ✅ Whisper base (Mac) で日本語 STT: 定常 0.3-1.5s
- ✅ OpenAI gpt-4o-mini で応答生成: 1-1.5s
- ✅ Robot 側 VAD → 送信 → 再生ループ
- ✅ Robot のマイク・スピーカー・DoA・mini.media API 全動作 (組立時のマイク FPC ケーブル逆挿し問題を解決済み)

### 解決済み (旧・停止点)
- ✅ **Style-Bert-VITS2 の numpy ABI エラーは解消** (2026-07-01)
  - 原因: `numpy>=1.26,<2.3` が numpy 2.x に解決され、`pyopenjtalk-dict` の C 拡張とABI不整合
  - 対処: `pyproject.toml` を **`numpy==1.26.4` にピン留め**（参考プロジェクト `../gakucho-ai-motion` と同じ）
  - 併せて日本語 BERT (`ku-nlp/deberta-v2-large-japanese-char-wwm`) を HF Hub から自動取得するよう
    `pc_server/sbv2_tts.py` に `_ensure_jp_bert()` を追加（初回のみDL）
  - SBV2 の infer は int16 を返すため、ラッパー側で float32[-1,1] に正規化する契約に統一

### Phase 1 で追加 (2026-07-01)
- ✅ **LangGraph ツール呼び出しエージェント** (`pc_server/agent/`) — gpt-4o-mini + ReAct ループ、
  会話履歴は MemorySaver、ツール: `get_current_time` / `get_weather` / `express_motion`
- ✅ **WebSocket 常時接続アーキテクチャ** (`pc_server/ws_server.py` `/ws/session`) — 文単位ストリーミングTTS
- ✅ **ロボット側 WS クライアント + Motion Executor** (`robot/conversation_ws.py`, `robot/motion.py`)
- 設計の下敷きは参考プロジェクト `../gakucho-ai-motion`（詳細は `.claude/plans/` の計画ファイル参照）

## 4. ファイル構成

```
~/reachy-mini-chat/
├── HANDOFF.md                   # この資料
├── pyproject.toml               # PC 側依存 (uv 管理)
├── uv.lock
├── .python-version              # 3.12
├── .env                         # OPENAI_API_KEY 等 (git 除外)
├── .env.example                 # テンプレ
├── .gitignore
│
├── pc_server/                   # PC 側 (macOS で動かす)
│   ├── __init__.py
│   ├── server.py                # FastAPI: /turn (POST) → WAV返却
│   └── sbv2_tts.py              # Style-Bert-VITS2 ラッパー (numpy ABI で停止中)
│
├── robot/                       # Robot 側 (/home/pollen/reachy-mini-chat/robot/ に rsync)
│   ├── conversation.py          # 本番: VAD→送信→再生ループ
│   ├── check_robot.py           # mini.media 疎通確認
│   ├── test_speaker.py          # 440Hz サイン波を鳴らす
│   ├── test_mic.py              # 5秒録音してエコーバック
│   ├── diagnose_mic.py          # マイクの生データ・DoA 診断
│   ├── stt_test.py              # faster-whisper で既存 WAV を文字起こし
│   └── stt_bench.py             # RPi での Whisper 各設定ベンチマーク
│
├── src/                         # PC 単体版 (Robotなしでも動く初期実装、参考実装)
│   ├── list_devices.py
│   ├── audio_devices.py
│   ├── record_and_transcribe.py # sounddevice + Whisper local
│   └── speak.py                 # macOS say + 出力デバイス指定
│
└── recordings/                  # 録音WAV (gitignore済み)
```

## 5. セットアップ手順 (新環境から動かす場合)

### 5.1 PC 側 (macOS Apple Silicon 前提)

```bash
cd ~/reachy-mini-chat
uv sync                  # .venv 作成 (Python 3.12)
cp .env.example .env
# .env を編集して OPENAI_API_KEY=sk-... を設定
```

**必要な追加準備 (Style-Bert-VITS2 を使う場合):**
- モデルファイル: `<SBV2 モデルのフォルダ>/` に `G_3700.pth`, `config.json`, `style_vectors.npy` を配置
- 別モデルを使う場合は `.env` の `SBV2_MODEL_DIR` で上書き

### 5.2 Robot 側 (Reachy Mini Wireless、事前セットアップ済み)

- SSH: `ssh pollen@reachy-mini.local` (鍵認証。PC の公開鍵を authorized_keys に登録して使う)
- Python 環境: `/venvs/apps_venv/bin/python` (Python 3.12、reachy_mini 1.7.3、faster-whisper, piper-tts, openai 導入済み)
- コード配置: `/home/pollen/reachy-mini-chat/`

**PC 側で変更したコードを Robot に反映するには:**
```bash
rsync -avz ~/reachy-mini-chat/robot/ \
      pollen@reachy-mini.local:/home/pollen/reachy-mini-chat/robot/
```

## 6. 動かし方 (完全手順)

### 6.1 PC でサーバー起動 (Terminal 1) — WebSocket 版 (現行)

```bash
cd ~/reachy-mini-chat
uv run uvicorn pc_server.ws_server:app --host 0.0.0.0 --port 8765
```

正常起動時のログ:
```
モデルをロード中… (Whisper / SBV2 / LangGraph agent)
[SBV2] loaded.
Ready. (4.7s)
INFO:     Uvicorn running on http://0.0.0.0:8765
```

- エンドポイント: `ws://<host>:8765/ws/session`（binary=発話PCM float32@16k、text(JSON)=stt/motion/turn_end）
- 旧 HTTP 版 `pc_server/server.py`（`/turn`）も残置（参考）。現行は WS 版を使う。

### 6.2 疎通確認 (Terminal 2)

```bash
curl http://<PC のホスト名>.local:8765/
# → JSON でモデル情報が返る
ssh pollen@reachy-mini.local 'curl -s http://<PC のホスト名>.local:8765/'
# → Robot からも同様に見えるはず
```

### 6.3 会話クライアント起動 (Terminal 2)

```bash
# 現行: WebSocket + Motion Executor 版
ssh pollen@reachy-mini.local \
  'PC_HOST=<PC のホスト名>.local /venvs/apps_venv/bin/python -u /home/pollen/reachy-mini-chat/robot/conversation_ws.py'
```

Robot に話しかけると、応答が文単位で逐次再生され、必要に応じて首/アンテナのモーションが再生される。
Ctrl+C で終了 (SSH 経由で残るときは `pkill -9 -f conversation_ws.py`)。
旧 HTTP 版は `robot/conversation.py`（`/turn` サーバーとセット）。

### 6.4 単体テスト用スクリプト

| 目的 | コマンド |
|---|---|
| Robot オーディオ疎通 | `ssh pollen@reachy-mini.local '/venvs/apps_venv/bin/python /home/pollen/reachy-mini-chat/robot/check_robot.py'` |
| スピーカーテスト | 同上 → `robot/test_speaker.py` |
| マイクテスト (ビープ後録音→エコー) | 同上 → `robot/test_mic.py --seconds 5` |
| マイク信号診断 | 同上 → `robot/diagnose_mic.py` |
| STT テスト (最新 WAV) | 同上 → `robot/stt_test.py --model tiny --language ja` |

## 7. 技術スタック と 選定理由

| 層 | 選定 | 理由 |
|---|---|---|
| STT (現) | OpenAI Whisper (`base`, PC 上) | RPi では遅すぎ。Mac Apple Silicon なら 0.3-1.5s |
| LLM (現) | OpenAI gpt-4o-mini | 会話用途に十分、レイテンシ 1-1.5s、コスト低 |
| TTS (旧) | macOS `say` (Kyoko) | 動作確認用の最小実装。無料・追加インストール不要 |
| TTS (目標) | Style-Bert-VITS2 + 学長モデル | ユーザー指定。声にキャラクター性を持たせる |
| VAD | mini.media.get_DoA().is_speech | Reachy 内蔵 XVF3800 のハードウェア VAD |
| 通信 | HTTP POST (raw float32 body) | シンプル。将来 WebSocket ストリームに拡張予定 |
| PC 環境管理 | uv + Python 3.12 (`<3.13` 固定) | 3.14 は pyopenjtalk 系がビルドできない |
| Robot 環境 | `/venvs/apps_venv/` (Reachy OS 標準) | 公式の Approach A に従う |

### API・接続情報

| 項目 | 値 |
|---|---|
| PC ホスト | `<PC のホスト名>.local` (mDNS) |
| Robot ホスト | `reachy-mini.local` (mDNS) |
| Robot SSH | `pollen@reachy-mini.local`、鍵認証 |
| サーバー port | 8765 |
| Robot daemon port | 8000 (reachy_mini API) |
| Robot Audio card | card 0 "Reachy Mini Audio" (VID:PID `38fb:1001`)、DFU 0212 |
| ALSA 論理デバイス | `plug:reachymini_audio_src` / `plug:reachymini_audio_sink` (dsnoop/dmix 経由) |

## 8. Reachy Mini SDK メモ (1.7.3)

- `ReachyMini()` は auto で local/network 検出。Robot 上ならそのまま
- `mini.media.*` API (16kHz float32 stereo が入出力):
  - `start_recording()` / `stop_recording()`
  - `start_playing()` / `stop_playing()`
  - `get_audio_sample()` → `(N, 2) float32` またはNone
  - `push_audio_sample(data)` → non-blocking (再生完了まで sleep 要)
  - `play_sound(path)` → 便利なファイル再生。任意 sr の WAV OK (内部でリサンプル)
  - `get_DoA()` → `(angle_rad, is_speech: bool)` またはNone
  - `get_input_audio_samplerate()` / `get_output_audio_samplerate()` (共に 16000)
  - `get_input_channels()` / `get_output_channels()` (共に 2)
- daemon がハードを常時握るので、`arecord -D hw:0,0` 等の直接アクセスはできない (busy)。dsnoop 経由なら共有可能
- daemon 制御:
  - `sudo systemctl stop|start reachy-mini-daemon`
  - `journalctl -u reachy-mini-daemon -f`
  - 健全性: `reachyminios_check` (ただし daemon 起動中は motor/audio/camera が false positive で fail 表示される)

## 9. 過去にハマった点 (再発防止)

### 9.1 マイクから音が拾えず全サンプル 0.0
**原因:** 頭部内の Mic Array Board (4x PDM MEMS + XMOS XVF3800) と Main Board を繋ぐ **FPC ケーブルの上下逆挿し**。組立時にほぼ確定で起きる。
**対処:**
- 電源 OFF → 頭部を開ける → FPC ケーブル向き確認
  - 白＋青ケーブル: 青面が上
  - 黒ケーブル: 「Main Board」文字面が上
- 参考画像: https://github.com/pollen-robotics/reachy_mini/raw/main/docs/assets/black_fpc_cable.png
- 公式トラブルシューティング: https://huggingface.co/docs/reachy_mini/troubleshooting の "Microphone not working" 節

### 9.2 uvicorn 起動時に startup 処理が走らない
**原因:** starlette 1.x で `@app.on_event("startup")` は無音で無効化される。
**対処:** `@asynccontextmanager` + `FastAPI(lifespan=lifespan)` に書き換え済み。

### 9.3 print が uvicorn ログに出ない
**原因:** stdout バッファリング。
**対処:** `flush=True, file=sys.stderr` の `_log()` ヘルパーで統一 (`pc_server/server.py`)。

### 9.4 SSH 越しの Ctrl+C で python プロセスが Robot に残る
**原因:** SSH の TTY 切断だけでは orphan になる。
**対処:** `ssh pollen@reachy-mini.local 'pkill -9 -f conversation.py'` で確実に殺す。

### 9.5 uv が Python 3.14 を選び pyopenjtalk がビルド不可
**対処:** `pyproject.toml` に `requires-python = ">=3.11,<3.13"` と `.python-version` に `3.12` を書いて固定済み。

### 9.6 openai-whisper が古い numba を選び llvmlite が 3.12 で失敗
**対処:** `numba>=0.59` を明示的に依存として入れて解決済み。

### 9.7 現在停止中: pyopenjtalk-dict の numpy ABI 不整合
セクション 3 の未解決課題を参照。

## 10. 次にやること (優先順)

### 10.1 【最優先】 SBV2 TTS を動かす
- Option A: `uv add "pyopenjtalk-dict==<stable版>"` で dev2 以外を試す
- Option B: `numpy<2` に完全ダウングレード (whisper 側と両立するか要検証)
- Option C: `pyopenjtalk-dict` を諦めて `pyopenjtalk` (無印) を強制インストール、style-bert-vits2 のコード側でパッチ
- Option D: Docker で Linux 環境を作って回避 (Mac の環境を汚さない副次効果あり)

### 10.2 会話体験の改善
- **STT 精度向上:** Whisper `base` → `small` or `medium` (レイテンシ +1-2s とのトレードオフ)
- **VAD 誤発火対策:** 最短発話時間を 0.4 → 0.8s、または最低RMS閾値追加
- **会話履歴管理:** 現状は無限累積。N 往復で古いのを切るか要約
- **フィードバック回避:** 現状は録音停止/再開で処理。XVF3800 の AEC を活かして全二重にする余地あり

### 10.3 ロボットらしさの追加
- 発話中に首・アンテナを微妙に動かす (`mini.goto_target()` + `create_head_pose()`)
- LLM の返答に「感情タグ」を含ませて、それに応じてモーションを選ぶ
- OpenAI Realtime API 版への移行 (音声in→音声out で低レイテンシ、ただし SBV2 の声を使えない)

### 10.4 運用面
- 起動を1コマンド化 (Makefile または `just` レシピ)
- systemd or launchd で自動起動 (常時稼働ロボットに)
- ウェイクワード対応 (「ねえ Reachy」で起動)

## 11. 参考リンク

- Reachy Mini 公式ドキュメント: https://huggingface.co/docs/reachy_mini/index
- Reachy Mini GitHub: https://github.com/pollen-robotics/reachy_mini
- Wireless セットアップ: https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/get_started
- 開発ワークフロー (Approach A): https://huggingface.co/docs/reachy_mini/platforms/reachy_mini/development_workflow
- SDK Python: https://huggingface.co/docs/reachy_mini/SDK/python-sdk
- トラブルシューティング FAQ: https://huggingface.co/docs/reachy_mini/troubleshooting
- Reachy Mini Discord (困ったとき): https://discord.gg/Y7FgMqHsub
- Style-Bert-VITS2: https://github.com/litagin02/Style-Bert-VITS2
- Piper TTS (代替候補): https://github.com/rhasspy/piper

## 12. 環境変数 (`.env`)

```dotenv
# 必須
OPENAI_API_KEY=sk-...

# 任意 (デフォルトあり)
WHISPER_MODEL=base                                    # tiny/base/small/medium/large
OPENAI_MODEL=gpt-4o-mini                              # gpt-4o, gpt-4o-mini 等
SBV2_MODEL_DIR=<SBV2 モデルのフォルダ> # G_*.pth の入ったフォルダ
SBV2_LENGTH=1.1                                       # 発話速度 (大きいほど遅い)
```

## 13. コミット状況

- ブランチ: `main`
- 最新 commit: `001d1a6` — "Initial: PC側 音声I/O実験 (sounddevice + Whisper + macOS say)"
- 未 commit: pc_server/*, robot/*, HANDOFF.md 等ほぼ全部

引き継ぎ後、動作確認 → まとめて commit する想定。SBV2 まわりが不安定なので、旧 say 実装を復元できるように branch を切って作業推奨。
