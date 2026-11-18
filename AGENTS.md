# AGENTS.md — 開発ルール

設計の正は [docs/DESIGN.md](docs/DESIGN.md)。経緯・実機セットアップは [HANDOFF.md](HANDOFF.md)。

## 開発サイクル（Issue 駆動）
1. GitHub Issue を 1 つ選ぶ（マイルストーンの若い順・P0 優先・依存が解決済みのもの）。着手時に自分を assign し、Issue にコメントで方針を書く。
2. `main` から `feat/<issue番号>-<短いslug>` ブランチを切る（CI/基盤は `ci/`、修正は `fix/`）。
3. 実装＋テスト。判断ロジックは純粋関数/クラスに分け、必ず単体テストを付ける。
4. ローカルで CI と同じチェックを通す:
   ```bash
   uv run ruff check . && uv run ruff format --check . && uv run pyright && uv run pytest
   ```
5. PR を作成（テンプレートに従う。本文に `Closes #<issue>`）。`needs-robot` の Issue は「実機確認」欄に手順を書き、オーナーに確認を依頼する。
6. **マージはオーナーが行う。** `main` へ直接 push しない。PR の CI が赤のままレビュー依頼しない。

## 守ること
- 1 PR = 1 Issue（小さく保つ）。無関係なリファクタを混ぜない。
- 役割分担を守る: 「いつ・誰に話すか」と人物の記憶は **ロボット**、STT・応答生成・TTS・会話履歴は **サーバー**。両者の間は `protocol/` の型だけで話す（設計書 §1.1, §2）。
- 発話タイミングは「空気を読む」。**タイマー式の自発発話は実装しない**（設計書 §3.4）。
- 秘密情報（API キー、パスワード、個人の顔データ `data/`）をコミットしない。
- CI に ML 依存（torch / whisper / SBV2）を持ち込まない。`pc-ml` グループに入れ、import は遅延させる。
- ロボット側コードを変えたら PR に rsync と実機確認手順を書く。

## よく使うコマンド
| 目的 | コマンド |
|---|---|
| PC 環境（全部入り） | `uv sync` |
| CI 相当の軽量環境 | `uv sync --no-default-groups --group dev` |
| PC サーバー起動 | `uv run uvicorn pc_server.ws_server:app --host 0.0.0.0 --port 8765` |
| ロボットへ反映 | `rsync -avz robot/ pollen@reachy-mini.local:/home/pollen/reachy-mini-chat/robot/` |
