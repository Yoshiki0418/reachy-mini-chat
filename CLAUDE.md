# CLAUDE.md — 開発ルール

設計の正は [docs/DESIGN.md](docs/DESIGN.md)。経緯・実機セットアップは [HANDOFF.md](HANDOFF.md)。

## リポジトリ構成（uv workspace）
| ディレクトリ | import 名 | 動く場所 | 役割 |
|---|---|---|---|
| `protocol/` | `reachy_protocol` | 両方 | ロボット⇄サーバー間のメッセージ定義（依存は pydantic のみ） |
| `robot_app/` | `reachy_front` | Reachy Mini（apps_venv） | 知覚・判断・人物の記憶・サーバー呼び出し・表現 |
| `server/` | `reachy_server` | 研究室 GPU マシン | STT・応答生成・TTS・会話履歴（ML 依存は `gpu` グループ） |
| `pc_server/`, `robot/` | — | — | 移行前の旧実装。新パッケージへ移し終えたら削除する |

新しいコードは新パッケージに書く。旧実装は移行の Issue でだけ触る。

## 開発サイクル（Issue 駆動）
1. GitHub Issue を 1 つ選ぶ（マイルストーンの若い順・P0 優先・依存が解決済みのもの）。着手時に自分を assign し、Issue にコメントで方針を書く。
2. `main` から `feat/<issue番号>-<短いslug>` ブランチを切る（CI/基盤は `ci/`、修正は `fix/`）。
3. 実装＋テスト。判断ロジックは純粋関数/クラスに分け、必ず単体テストを付ける。
4. ローカルで CI と同じチェックを通す（pre-commit を入れていればコミット時に lint・秘密情報チェック、push 時に pyright が自動で走る）:
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
| Git フックの導入（初回のみ） | `uv run pre-commit install` |
| フックを全ファイルに実行 | `uv run pre-commit run --all-files` |
| CI と同じ環境でチェック（ML 依存なし） | `export UV_PROJECT_ENVIRONMENT=.venv-ci && uv sync --no-default-groups --group dev && uv run pyright && uv run pytest` |
| 旧 PC サーバー起動 | `uv run uvicorn pc_server.ws_server:app --host 0.0.0.0 --port 8765` |
| 新サーバー起動（GPU マシン） | `uv run uvicorn reachy_server.app:create_app --factory --host 0.0.0.0 --port 8765`（`REACHY_SERVER_TOKEN` 必須） |
| ロボットへ反映 | `rsync -avz robot/ pollen@reachy-mini.local:/home/pollen/reachy-mini-chat/robot/` |
