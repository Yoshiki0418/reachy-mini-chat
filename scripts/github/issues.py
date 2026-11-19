"""Issue 定義（設計書 改訂 2: ロボット＝フロントエンド / GPU サーバー＝生成バックエンド）。

各 Epic = マイルストーン 1 つ。tasks が sub-issue になる。
"""

LABELS = {
    "epic": ("5319e7", "親 Issue（マイルストーン単位）"),
    "type:feature": ("1d76db", "機能追加"),
    "type:chore": ("c5def5", "整備・リファクタ"),
    "type:ci": ("0e8a16", "CI / 開発基盤"),
    "type:research": ("fbca04", "調査・ベンチマーク"),
    "type:bug": ("d73a4a", "不具合"),
    "side:robot": ("006b75", "ロボット（フロントエンド）"),
    "side:server": ("5a32a3", "GPU サーバー（生成バックエンド）"),
    "side:protocol": ("0052cc", "ロボット⇄サーバー間プロトコル"),
    "area:harness": ("bfd4f2", "core / harness"),
    "area:perception": ("bfd4f2", "知覚（音声・視覚）"),
    "area:social": ("bfd4f2", "宛先判定・自発発話"),
    "area:memory": ("bfd4f2", "人物メモリ"),
    "area:agent": ("bfd4f2", "LangGraph エージェント"),
    "area:expression": ("bfd4f2", "音声再生・モーション"),
    "area:ops": ("bfd4f2", "デプロイ・運用"),
    "needs-robot": ("d93f0b", "実機での確認が必要"),
    "needs-gpu": ("d93f0b", "GPU サーバーでの確認が必要"),
    "P0": ("b60205", "最優先"),
    "P1": ("d93f0b", "高"),
    "P2": ("fbca04", "中"),
}


def body(bg: str, todo: list[str], ac: list[str], deps: str = "なし", ref: str = "") -> str:
    t = "\n".join(f"- [ ] {x}" for x in todo)
    a = "\n".join(f"- [ ] {x}" for x in ac)
    r = f"`docs/DESIGN.md` {ref}".strip()
    return (
        f"## 背景\n{bg}\n\n## やること\n{t}\n\n## 受け入れ条件\n{a}\n\n## 依存\n{deps}\n\n参照: {r}"
    )


R, S, P = "side:robot", "side:server", "side:protocol"

EPICS = [
    # ------------------------------------------------------------------ M0
    {
        "milestone": "M0 基盤整備",
        "title": "[Epic] M0 基盤整備: main 取り込み・CI・テスト基盤",
        "desc": "現行コードを main に取り込み、PR ごとに品質を自動確認できる状態にする。設計書 §7。",
        "tasks": [
            (
                "現行実装 (Phase 0〜5) と設計書を main に取り込む",
                ["type:chore", "P0"],
                body(
                    "main は初期コミットのみ。Phase 0〜5 は feature/langgraph-agent にある。",
                    ["feature/langgraph-agent → main の PR を作成しマージ"],
                    ["main に現行実装と docs/DESIGN.md が入っている"],
                ),
            ),
            (
                "依存グループの分割と CI（lock / lint / typecheck / test）",
                ["type:ci", "P0"],
                body(
                    "main マージ前に問題を把握したい。CI に ML 依存を入れない。",
                    [
                        "pyproject を core / pc-ml / dev に分割",
                        "GitHub Actions: uv lock --check, ruff, pyright, pytest + カバレッジ",
                        "集約ジョブ CI OK",
                    ],
                    ["PR に CI ステータスが付く", "わざと壊した変更で失敗する"],
                    ref="§7",
                ),
            ),
            (
                "テスト基盤と既存ロジックの単体テスト",
                ["type:ci", "P0"],
                body(
                    "テストがゼロだった。応答ゲート・自発発話ポリシー・人物メモリから始める。",
                    [
                        "tests/ と conftest（reachy_mini SDK のスタブ）",
                        "Fake STT/TTS/Agent",
                        "応答ゲート・ProactivePolicy・PersonStore・ツールのテスト",
                    ],
                    ["実機・API キーなしで CI が通る"],
                ),
            ),
            (
                "PR / Issue テンプレート・開発ルール・ブランチ保護",
                ["type:ci", "P1"],
                body(
                    "Claude が PR を出しオーナーがレビューする運用を定型化する。",
                    [
                        "PR テンプレート（実機確認欄つき）",
                        "Issue テンプレート",
                        "CLAUDE.md / AGENTS.md",
                        "main 保護（CI OK 必須・PR 必須）",
                    ],
                    ["CI が赤だとマージできない"],
                ),
            ),
            (
                "pre-commit フック",
                ["type:chore", "P2"],
                body(
                    "ローカルで CI と同じチェックを素早く回す。",
                    ["pre-commit（ruff）"],
                    ["`pre-commit run -a` が通る"],
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M1
    {
        "milestone": "M1 プロトコルとリポジトリ再編",
        "title": "[Epic] M1 プロトコルとリポジトリ再編: uv workspace・protocol v1・契約テスト",
        "desc": "ロボットとサーバーの間の契約を先に固め、両側を独立して開発できるようにする。設計書 §2, §6。",
        "tasks": [
            (
                "uv workspace 化（protocol / robot_app / server）",
                ["type:chore", P, "P0"],
                body(
                    "ロボット・サーバー・共有プロトコルを 1 リポジトリで分けて管理する。",
                    [
                        "ルート pyproject を workspace 化",
                        "protocol/, robot_app/, server/ の各 pyproject（server の ML 依存は gpu グループ）",
                        "既存 pc_server / robot は当面そのまま動かす",
                    ],
                    ["`uv sync` で全パッケージが入る", "既存サーバーが従来どおり起動する"],
                    ref="§6",
                ),
            ),
            (
                "パッケージ別 CI（test-protocol / test-robot / test-server）",
                ["type:ci", "P0"],
                body(
                    "workspace 化に合わせて CI を分ける。",
                    [
                        "パッケージごとの pytest ジョブ",
                        "変更パスに応じたスキップは行わず全ジョブ実行（必須チェックを安定させる）",
                        "CI OK に集約",
                    ],
                    ["各パッケージのテストが独立に走る"],
                    "uv workspace 化",
                    "§7",
                ),
            ),
            (
                "protocol v1: メッセージモデル定義",
                ["type:feature", P, "P0"],
                body(
                    "設計書 §2.3 の全メッセージを pydantic で定義する。",
                    [
                        "hello / welcome / utterance.* / transcript / floor.decision / proactive.request / context",
                        "turn.* / sentence / motion / memory.op / memory.ack / tool.call / tool.result / turn.cancel / person.forget / session.close / ping / error",
                        "discriminated union で parse",
                        "protocol_version と互換判定",
                    ],
                    ["全メッセージの往復（serialize → parse）テスト", "版の互換判定テスト"],
                    "uv workspace 化",
                    "§2",
                ),
            ),
            (
                "protocol: バイナリフレーミング（audio_in / audio_out / frame）",
                ["type:feature", P, "P0"],
                body(
                    "[ヘッダ長][JSON ヘッダ][ペイロード] 形式。",
                    [
                        "encode / decode",
                        "PCM float32 / WAV PCM16 / JPEG の format 定義",
                        "不正フレームの扱い",
                    ],
                    ["境界値・破損データのテスト"],
                    "protocol v1 メッセージ",
                    "§2.1",
                ),
            ),
            (
                "契約テストの枠組み（FakeServer / FakeRobotClient）",
                ["type:ci", P, "P1"],
                body(
                    "片側だけ変えて相手を壊すのを CI で防ぐ。",
                    [
                        "プロトコル準拠の FakeServer（ロボット側テスト用）",
                        "FakeRobotClient（サーバー側テスト用）",
                        "シナリオ: 1 ターン応答 / ignore / 自発発話 / 取消 / memory.op",
                    ],
                    ["tests/contract が CI で走る"],
                    "protocol v1 メッセージ",
                ),
            ),
            (
                "プロトコル仕様ドキュメントの自動生成",
                ["type:chore", P, "P2"],
                body(
                    "モデルから JSON Schema と Markdown の仕様書を生成する。",
                    ["docs/protocol.md を生成するスクリプト", "CI で生成物の差分を検査"],
                    ["仕様書とコードがずれない"],
                    "protocol v1 メッセージ",
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M2
    {
        "milestone": "M2 サーバー",
        "title": "[Epic] M2 サーバー: 研究室 GPU マシン上の生成バックエンド",
        "desc": "STT・応答生成・TTS・会話履歴を担う自作サーバーを作る。設計書 §4。",
        "tasks": [
            (
                "GPU マシンのベンチマーク（faster-whisper / SBV2 / VRAM）",
                ["type:research", S, "needs-gpu", "P0"],
                body(
                    "モデルの選択とレイテンシ予算を数値で決める。",
                    [
                        "faster-whisper の各サイズ（small / medium / large-v3-turbo）の速度と日本語精度",
                        "SBV2 の 1 文あたりの合成時間",
                        "同時常駐時の VRAM",
                        "docs/benchmarks.md に記録",
                    ],
                    ["STT・TTS のモデルを決められる"],
                ),
            ),
            (
                "サーバー骨格: FastAPI・設定・/healthz・WS /v1/session・認証",
                ["type:feature", S, "P0"],
                body(
                    "新しいサーバーの土台。",
                    [
                        "server/ パッケージ、pydantic-settings",
                        "hello / welcome のハンドシェイク（トークン認証・版チェック）",
                        "ping / pong、/healthz",
                    ],
                    ["FakeRobotClient でハンドシェイクのテスト"],
                    "protocol v1 メッセージ",
                    "§4.1",
                ),
            ),
            (
                "セッションとターンの状態機械",
                ["type:feature", S, "P0"],
                body(
                    "transcribing → waiting_decision → generating → done / cancelled。",
                    [
                        "utt_id / turn_id の管理",
                        "decision 待ちのタイムアウト",
                        "turn.cancel で生成と合成を止める",
                    ],
                    ["状態遷移の単体テスト（取消・タイムアウト含む）"],
                    "サーバー骨格",
                ),
            ),
            (
                "Provider 抽象と registry（STT / LLM / TTS）と Fake",
                ["type:feature", S, "area:harness", "P0"],
                body(
                    "GPU なしの CI と、本番の GPU 構成を切り替える。",
                    [
                        "Protocol 定義と registry",
                        "FakeSTT / FakeLLM / FakeTTS",
                        "設定でプロバイダを選ぶ",
                    ],
                    ["切替の単体テスト"],
                    "サーバー骨格",
                ),
            ),
            (
                "STT プロバイダ: faster-whisper（GPU）とストリーミング入力",
                ["type:feature", S, "area:perception", "needs-gpu", "P0"],
                body(
                    "発話中に届く音声を逐次処理し、utterance.end から transcript までを短くする。",
                    [
                        "faster-whisper プロバイダ（遅延 import、gpu グループ）",
                        "audio_in の逐次バッファリング",
                        "transcript の送信",
                    ],
                    ["Fake でのテスト", "GPU で 0.5 s 以下"],
                    "Provider 抽象, GPU ベンチ",
                ),
            ),
            (
                "TTS プロバイダ: SBV2（GPU）と文単位ストリーミング",
                ["type:feature", S, "area:expression", "needs-gpu", "P0"],
                body(
                    "既存 pc_server/sbv2_tts.py を移植する。",
                    [
                        "SBV2 プロバイダ（GPU 推論）",
                        "文ごとに合成して audio_out を送信",
                        "取消時に待ち行列を破棄",
                    ],
                    ["Fake でのテスト", "GPU で 1 文 0.3 s 以下"],
                    "Provider 抽象, GPU ベンチ",
                ),
            ),
            (
                "既存エージェントのサーバーへの移植と context 受け取り",
                ["type:feature", S, "area:agent", "P0"],
                body(
                    "pc_server の LangGraph エージェントを server/ に移し、ロボットから届く context を使う。",
                    [
                        "floor.decision=respond / proactive.request で生成開始",
                        "context.person / situation をプロンプトに反映",
                        "sentence / audio_out / turn.end の送信",
                    ],
                    ["Fake LLM で 1 ターンの契約テスト"],
                    "セッションとターンの状態機械, TTS プロバイダ",
                ),
            ),
            (
                "会話履歴ストア（thread = robot_id:person_id:date）と場の文脈バッファ",
                ["type:feature", S, "area:memory", "P0"],
                body(
                    "サーバー側で会話履歴を持つ（オーナー方針）。",
                    [
                        "SqliteSaver による永続化",
                        "ignore された発話は短期バッファ（既定 10 分・非永続）",
                        "person.forget で該当スレッドを削除",
                    ],
                    ["削除と保持期限の単体テスト"],
                    "既存エージェントの移植",
                ),
            ),
            (
                "記憶ツールを memory.op 発行に置き換える",
                ["type:feature", S, "area:memory", "P0"],
                body(
                    "人物の記憶の正はロボット。サーバーは操作を発行するだけにする。",
                    [
                        "set_person_name / remember_fact / forget_me → memory.op",
                        "memory.ack の rejected を以後の文脈に使わない",
                    ],
                    ["契約テスト"],
                    "既存エージェントの移植, protocol v1 メッセージ",
                ),
            ),
            (
                "ロボット側ツールの RPC（tool.call / tool.result）",
                ["type:feature", S, P, "P1"],
                body(
                    "lab_presence などロボットのデータを使うツールのため。",
                    [
                        "tool.call の発行と結果待ち（タイムアウト 1 s）",
                        "タイムアウト時に LLM へ返すメッセージ",
                    ],
                    ["契約テスト"],
                    "既存エージェントの移植",
                ),
            ),
            (
                "会話要約の生成（session.close → memory.op add_episode）",
                ["type:feature", S, "area:memory", "P1"],
                body(
                    "相手が退出したら会話を要約し、ロボットに保存させる。",
                    ["要約プロンプト", "add_episode の発行"],
                    ["Fake LLM でのテスト"],
                    "会話履歴ストア",
                ),
            ),
            (
                "宛先判定の補助 API（/v1/classify）",
                ["type:feature", S, "area:social", "P2"],
                body(
                    "ロボットのルールで決めきれない発話を LLM で分類する。",
                    ["JSON 出力の分類プロンプト", "400 ms のタイムアウト"],
                    ["Fake LLM でのテスト"],
                    "Provider 抽象",
                ),
            ),
            (
                "フィラーと定型挨拶の事前合成・配布",
                ["type:feature", S, "area:expression", "P1"],
                body(
                    "体感遅延の短縮と、サーバー障害時のフォールバック用。",
                    ["起動時にフィラーと定型挨拶を合成", "ロボットからの要求で配布（版つき）"],
                    ["ロボット側でキャッシュできる"],
                    "TTS プロバイダ",
                ),
            ),
            (
                "トレースとレイテンシ計測（サーバー）",
                ["type:feature", S, "area:harness", "P1"],
                body(
                    "どこが遅いかを追えるようにする。",
                    [
                        "ターン単位の区間時間（受信→STT→LLM 初回トークン→TTS 初回）",
                        "JSON ログ、LangSmith は任意で有効化",
                    ],
                    ["1 ターンのログから内訳が読める"],
                    "セッションとターンの状態機械",
                ),
            ),
            (
                "サーバーのデプロイ（systemd / Docker Compose）",
                ["type:chore", S, "area:ops", "needs-gpu", "P1"],
                body(
                    "研究室 GPU マシンで常駐させる。",
                    [
                        "deploy/server/ のユニットまたは compose",
                        "モデルの事前ダウンロード",
                        "秘密情報は .env（サーバーのみ）",
                    ],
                    ["再起動後に自動で起動する"],
                    "サーバー骨格",
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M3
    {
        "milestone": "M3 ロボットのフロントエンド基盤",
        "title": "[Epic] M3 ロボットのフロントエンド基盤: 骨格・イベントバス・サーバークライアント",
        "desc": "ロボットを「知覚・判断・記憶・API 呼び出し・表現」のフロントエンドとして作り直す。設計書 §3.1。",
        "tasks": [
            (
                "robot_app 骨格と設定（pydantic-settings）",
                ["type:feature", R, "area:harness", "P0"],
                body(
                    "既存 robot/ の env しきい値（EC_*, PROACTIVE_*, GATE_*）を集約する。",
                    [
                        "robot_app/{core,io,perception,world,social,memory,client,expression,harness}",
                        "Settings と起動時のダンプ",
                        "apps_venv へのインストール手順",
                    ],
                    ["設定の単体テスト", "既存の env 名と互換"],
                    "uv workspace 化",
                ),
            ),
            (
                "型付きイベントとイベントバス",
                ["type:feature", R, "area:harness", "P0"],
                body(
                    "モジュール間を疎結合にし、全イベントを記録できるようにする。",
                    [
                        "設計書 §3.2 のイベントを pydantic で定義",
                        "asyncio pub/sub（購読者の例外を隔離）",
                        "trace_id",
                    ],
                    ["単体テスト"],
                    "robot_app 骨格",
                ),
            ),
            (
                "RobotIO 抽象と ReachyRobotIO / FakeRobotIO",
                ["type:feature", R, "area:harness", "P0"],
                body(
                    "SDK 依存を 1 か所に閉じ込め、CI で実機なしに動かす。",
                    [
                        "Protocol（マイク・DoA・カメラ・スピーカー・首・アンテナ）",
                        "既存 robot/ のコードを ReachyRobotIO に移植",
                        "FakeRobotIO（WAV・画像列の再生、送ったコマンドの記録）",
                    ],
                    ["FakeRobotIO を使ったテスト"],
                    "robot_app 骨格",
                ),
            ),
            (
                "World Model（イベントの畳み込みとスナップショット）",
                ["type:feature", R, "area:harness", "P0"],
                body(
                    "判断モジュールが参照する「いまの状況」。",
                    [
                        "TrackState / RobotState / ConversationState",
                        "reducer",
                        "読み取り専用スナップショット",
                    ],
                    ["reducer の単体テスト"],
                    "イベントバス",
                ),
            ),
            (
                "サーバークライアント（接続・再接続・送受信・取消）",
                ["type:feature", R, P, "P0"],
                body(
                    "ロボットからサーバー API を呼ぶ部分。",
                    [
                        "hello / welcome、ハートビート、指数バックオフの再接続",
                        "発話中の audio_in ストリーミング送信",
                        "sentence / audio_out の受信キュー、turn.cancel",
                    ],
                    ["FakeServer との契約テスト"],
                    "protocol v1 メッセージ, イベントバス",
                ),
            ),
            (
                "音声再生キューとターンの結線（旧構成と同等の会話）",
                ["type:feature", R, "area:expression", "needs-robot", "P0"],
                body(
                    "新しいサーバー＋新しいフロントエンドで、まず今と同じ会話ができるようにする。",
                    [
                        "VAD → utterance.* → transcript → 既存ゲート相当の判定 → 再生",
                        "再生中の録音停止とエコー待ち（既存ロジックを移植）",
                    ],
                    ["実機で会話できる（オーナー確認）"],
                    "サーバークライアント, RobotIO 抽象",
                ),
            ),
            (
                "サーバー断のフォールバック",
                ["type:feature", R, "area:harness", "P1"],
                body(
                    "サーバーが落ちてもロボットは黙って固まらない。",
                    [
                        "ServerConnectionChanged で縮退モードへ",
                        "キャッシュした定型音声とモーションで反応",
                        "復帰後は通常動作",
                    ],
                    ["障害注入テスト"],
                    "サーバークライアント",
                ),
            ),
            (
                "Session Recorder / Replay（ロボット）",
                ["type:feature", R, "area:harness", "P1"],
                body(
                    "実機で起きたことをオフラインで再現して回帰テストする。",
                    [
                        "全イベントを JSONL に記録（音声・縮小フレームは任意）",
                        "判断層に再投入して判定の差分を出す",
                        "保持期限",
                    ],
                    ["記録して再生すると同じ判定になるテスト"],
                    "イベントバス, World Model",
                ),
            ),
            (
                "ロボットの systemd ユニット・ウォッチドッグ・デプロイスクリプト",
                ["type:chore", R, "area:ops", "needs-robot", "P1"],
                body(
                    "電源を入れたら自動で起動し、落ちても復帰する。",
                    [
                        "deploy/robot/reachy-front.service",
                        "deploy.sh（rsync + インストール + 再起動）",
                        "起動ログに git SHA",
                    ],
                    ["プロセスを kill しても自動で復帰する"],
                    "音声再生キューとターンの結線",
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M4
    {
        "milestone": "M4 知覚",
        "title": "[Epic] M4 知覚: 多人数トラッキング・顔同定・視線・入室検知",
        "desc": "誰が・何人・どこを見て・入ってきたのか通り過ぎたのかを、ロボット上で把握する。設計書 §3。",
        "tasks": [
            (
                "CM4 ベンチマーク（YuNet / SFace / 同時実行の余裕）",
                ["type:research", R, "needs-robot", "P0"],
                body(
                    "ロボット上の処理配分を数値で決める。",
                    [
                        "YuNet 320px の fps、SFace 1 顔あたりの ms、CPU と温度",
                        "音声・モーション・視覚の同時実行",
                    ],
                    ["docs/benchmarks.md に記録"],
                ),
            ),
            (
                "多人数フェイストラッカー（track_id）",
                ["type:feature", R, "area:perception", "P0"],
                body(
                    "今は最も大きい顔 1 つだけを見ている。",
                    [
                        "IoU と中心距離によるトラッキング",
                        "FaceTracked イベント",
                        "純粋ロジックを分離",
                    ],
                    ["合成データでの単体テスト"],
                    "イベントバス",
                ),
            ),
            (
                "ロボット上の顔同定（SFace）とトラック内の多数決",
                ["type:feature", R, "area:perception", "area:memory", "P0"],
                body(
                    "Phase 5 の同定は PC 側にある。ロボットへ移し、生体情報を外に出さない。",
                    [
                        "SFace の埋め込みを数 Hz で計算",
                        "トラック内の多数決で person_id を確定",
                        "PersonIdentified イベント",
                    ],
                    ["既知・未知の判定テスト"],
                    "多人数トラッカー, CM4 ベンチ",
                ),
            ),
            (
                "視線（アイコンタクト）推定の移植とトラック単位化",
                ["type:chore", R, "area:perception", "P1"],
                body(
                    "既存 GazeTracker の判定（nose_ratio / level / vert / face_frac）を純粋関数にする。",
                    [
                        "frontal 判定関数とテスト",
                        "トラックごとのアイコンタクト継続時間",
                        "発話区間の視線率を utterance.end の features に入れる",
                    ],
                    ["既存のしきい値で同等に判定する"],
                    "多人数トラッカー",
                ),
            ),
            (
                "DoA と顔の位置の対応付け（話者推定）",
                ["type:feature", R, "area:perception", "needs-robot", "P1"],
                body(
                    "複数人がいるとき、誰が話しているかを推定する。",
                    ["DoA の角度とカメラの水平位置のキャリブレーション", "speaking_track の推定"],
                    ["実機で 2 人の話者を区別できる"],
                    "多人数トラッカー",
                ),
            ),
            (
                "入室・通過・退出の検知",
                ["type:feature", R, "area:perception", "needs-robot", "P0"],
                body(
                    "受付として最も重要なきっかけ。",
                    [
                        "顔サイズの推移・滞在時間・移動方向から PersonEntered / PassedBy / Left を判定",
                        "入口の向き（設置方向）を設定化",
                        "記録データでの再生テスト",
                    ],
                    ["実機で入室と通過を区別できる（誤検知率を記録）"],
                    "多人数トラッカー, Recorder / Replay",
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M5
    {
        "milestone": "M5 記憶",
        "title": "[Epic] M5 記憶: SQLite・来訪ログ・親密度・memory.op・忘却",
        "desc": "来た人をロボットが覚え、来訪頻度で話しかけ方を変える。設計書 §3.5, §3.6。",
        "tasks": [
            (
                "人物メモリの SQLite 化と people.json からの移行",
                ["type:feature", R, "area:memory", "P0"],
                body(
                    "JSON では来訪ログや検索に向かない。",
                    [
                        "スキーマ（persons / face_embeddings / visits / facts / episodes）",
                        "マイグレーション機構",
                        "people.json の移行スクリプト",
                    ],
                    ["移行の単体テスト"],
                    "robot_app 骨格",
                ),
            ),
            (
                "来訪ログの記録と来訪の確定",
                ["type:feature", R, "area:memory", "P0"],
                body(
                    "入室イベントから来訪を確定して記録する。",
                    ["PersonEntered / Left → visits", "再入室の扱い（T 分以内は同じ来訪）"],
                    ["境界条件の単体テスト"],
                    "SQLite 化, 入室検知",
                ),
            ),
            (
                "親密度区分（familiarity）の算出",
                ["type:feature", R, "area:memory", "P0"],
                body(
                    "first_time / new / daily / regular / long_absence / returning_today。",
                    ["来訪履歴から区分・連続日数・不在日数を出す純粋関数", "境界値を設定化"],
                    ["区分ごとの単体テスト"],
                    "来訪ログ",
                ),
            ),
            (
                "プロフィールと状況の構築（context の組み立て）",
                ["type:feature", R, "area:memory", P, "P0"],
                body(
                    "毎ターン、サーバーへ送る context.person / situation を作る。",
                    [
                        "名前・親密度・連続日数・事実（上限・鮮度順）・直近の要約",
                        "人数・時間帯・曜日・同席者",
                    ],
                    ["単体テスト"],
                    "親密度区分, protocol v1 メッセージ",
                ),
            ),
            (
                "memory.op の検証と適用（memory.ack）",
                ["type:feature", R, "area:memory", "P0"],
                body(
                    "サーバーの LLM が出した記憶操作を、ロボットが検証してから保存する。",
                    [
                        "名前らしさ・長さ・重複・機微情報のブロックリスト",
                        "add_episode の保存",
                        "ack の返送",
                    ],
                    ["検証ルールの単体テスト", "契約テスト"],
                    "SQLite 化, サーバークライアント",
                ),
            ),
            (
                "ロボット側ツール: lab_presence / recall_person",
                ["type:feature", R, "area:memory", "P1"],
                body(
                    "サーバーからの tool.call に来訪ログで答える。",
                    ["今日来た人（同意の範囲内）", "人物の想起"],
                    ["単体テスト"],
                    "来訪ログ, ロボット側ツールの RPC",
                ),
            ),
            (
                "事実メモリの整理（重複の統合・上限・鮮度）",
                ["type:feature", R, "area:memory", "P2"],
                body(
                    "remember_fact が増え続ける問題。",
                    ["重複・矛盾の統合", "1 人あたりの上限と last_used_at による選択"],
                    ["単体テスト"],
                    "SQLite 化",
                ),
            ),
            (
                "プライバシー: 忘却・未命名人物の自動削除・掲示文",
                ["type:feature", R, "area:memory", "P1"],
                body(
                    "顔を覚えるシステムとして必須。",
                    [
                        "forget_me でロボットの全データを削除し person.forget をサーバーへ送る",
                        "未命名人物を N 日で自動削除",
                        "研究室に掲示する説明文 docs/privacy.md",
                    ],
                    ["削除の単体テストと契約テスト"],
                    "SQLite 化, 会話履歴ストア",
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M6
    {
        "milestone": "M6 社会的判断",
        "title": "[Epic] M6 社会的判断: 宛先判定・自発発話・割り込み・相槌",
        "desc": "研究室の一員として「話す番か」「自分宛てか」をロボットが判断する。タイマー式は禁止。設計書 §3.3, §3.4。",
        "tasks": [
            (
                "シナリオ評価フレームワーク（YAML → 期待する判定）",
                ["type:ci", R, "area:social", "P0"],
                body(
                    "判断層を CI で決定的に検証する。",
                    [
                        "シナリオ形式（World スナップショット＋発話＋期待する判定）",
                        "pytest ランナー",
                        "初期シナリオ 20 件（1 人 / 複数人 / 名前呼び / 人同士の会話 / 独り言）",
                    ],
                    ["CI で全件実行"],
                    "World Model",
                ),
            ),
            (
                "FloorManager: ルールベースの宛先判定",
                ["type:feature", R, "area:social", "P0"],
                body(
                    "既存の応答ゲート（eye_contact か名前呼び）を、特徴量のスコアに一般化する。",
                    [
                        "特徴量（名前呼び・発話中の視線率・話者＝注視者・人数・会話の継続・人同士の会話）",
                        "RESPOND / BACKCHANNEL / IGNORE と理由",
                        "floor.decision の送信",
                    ],
                    ["シナリオ評価が全件通る"],
                    "シナリオ評価, World Model",
                ),
            ),
            (
                "FloorManager: 曖昧なときの分類 API 利用",
                ["type:feature", R, "area:social", "P2"],
                body(
                    "ルールで決めきれないときだけサーバーに尋ねる。",
                    [
                        "スコアが曖昧帯のときに /v1/classify を呼ぶ",
                        "タイムアウトや失敗時は IGNORE 側に倒す",
                    ],
                    ["FakeServer でのテスト"],
                    "ルールベースの宛先判定, 宛先判定の補助 API",
                ),
            ),
            (
                "InitiativePolicy: 入室挨拶・アイコンタクト・見送り（タイマー禁止）",
                ["type:feature", R, "area:social", "P0"],
                body(
                    "既存 ProactivePolicy を一般化し、入室のきっかけと親密度別の戦略を加える。",
                    [
                        "入室・持続アイコンタクト・見送りのきっかけ",
                        "抑制（発話中・人同士の会話中・MIN_GAP・入室挨拶は 1 日 1 回）",
                        "proactive.request（strategy つき）",
                    ],
                    ["シナリオ評価で抑制条件を検証"],
                    "入室検知, 親密度区分",
                ),
            ),
            (
                "BargeIn: ロボットの発話中の割り込みへの対応",
                ["type:feature", R, "area:social", "area:expression", "needs-robot", "P1"],
                body(
                    "話している途中に話しかけられたら止まって聞く。",
                    [
                        "発話中の VAD 検出（自分の声のエコーと区別する）",
                        "再生とモーションの停止、turn.cancel",
                    ],
                    ["実機で途中で止まれる"],
                    "サーバークライアント",
                ),
            ),
            (
                "聞いている間の相槌（うなずき）",
                ["type:feature", R, "area:expression", "P1"],
                body(
                    "BACKCHANNEL の判定と、ユーザーが話している間のうなずき。",
                    ["SpeechEnded 直後の小さなうなずき", "BACKCHANNEL のときの非言語反応"],
                    ["実機で自然に見える（オーナー確認）"],
                    "ルールベースの宛先判定",
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M7
    {
        "milestone": "M7 エージェント刷新",
        "title": "[Epic] M7 エージェント刷新: StateGraph・構造化コンテキスト・受付嬢ペルソナ",
        "desc": "サーバー上のエージェント本体を、研究室の一員として振る舞うように作り直す。設計書 §4.2。",
        "tasks": [
            (
                "create_react_agent から明示的な StateGraph へ移行",
                ["type:feature", S, "area:agent", "P0"],
                body(
                    "コンテキスト構築やガードのノードを足すため。",
                    ["build_context → respond ⇄ tools → guard → emit", "AgentState の定義"],
                    ["Fake LLM でのグラフ遷移テスト"],
                    "既存エージェントの移植",
                ),
            ),
            (
                "構造化コンテキストの注入",
                ["type:feature", S, "area:agent", "P0"],
                body(
                    "「（相手メモ：…）」をユーザー発話の前に付ける方式をやめる。",
                    [
                        "context.person / situation / 自発発話の意図をシステムメッセージに構造化して入れる"
                    ],
                    ["プロンプト組み立ての単体テスト"],
                    "StateGraph",
                ),
            ),
            (
                "受付嬢ペルソナと親密度別の挨拶戦略",
                ["type:feature", S, "area:agent", "P0"],
                body(
                    "研究室に来るモチベーションを高める話しかけ方。",
                    [
                        "prompts/ を persona / style / greeting / safety に分割",
                        "親密度・時間帯・曜日・連続日数ごとの戦略",
                        "責める口調の禁止",
                    ],
                    ["LLM 評価シナリオで確認"],
                    "構造化コンテキスト",
                ),
            ),
            (
                "ガードノード（長さ・禁止事項・読み上げの正規化）",
                ["type:feature", S, "area:agent", "P1"],
                body(
                    "音声出力に向かない応答や不適切な発言を防ぐ。",
                    [
                        "文数の上限",
                        "容姿の評価・機微情報の詮索・責める口調の検出",
                        "記号の読み上げ正規化",
                    ],
                    ["単体テスト"],
                    "StateGraph",
                ),
            ),
            (
                "感情タグつきストリーミング出力",
                ["type:feature", S, "area:agent", "P1"],
                body(
                    "文ごとの感情を sentence.emotion でロボットに渡す。",
                    ["`<emo:...>` タグのストリーミングパーサ", "タグがないときのフォールバック"],
                    ["パーサの単体テスト"],
                    "StateGraph",
                ),
            ),
            (
                "VLM による場面理解（縮小フレーム）",
                ["type:feature", S, "area:agent", "P2"],
                body(
                    "必要なときだけフレームを送り、見えたものに触れる。",
                    ["frame（binary）の受信と LLM への添付", "送信の頻度をロボット側で制御"],
                    ["Fake でのテスト"],
                    "構造化コンテキスト",
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M8
    {
        "milestone": "M8 表現",
        "title": "[Epic] M8 表現: 発話と同期したモーション・フィラー",
        "desc": "対話内容に合わせて体を動かす。設計書 §3.1 expression。",
        "tasks": [
            (
                "ExpressionPlanner: 感情タグ → モーション",
                ["type:feature", R, "area:expression", "P0"],
                body(
                    "LLM のツール呼び出しに頼らず、文ごとに動作を付ける。",
                    ["感情とモーションの対応（設定化）", "文の再生開始に合わせて発火"],
                    ["単体テスト", "実機で同期を確認"],
                    "感情タグつき出力",
                ),
            ),
            (
                "モーション合成（反射・ジェスチャ・ベースの優先度）",
                ["type:feature", R, "area:expression", "needs-robot", "P1"],
                body(
                    "視線追従・ジェスチャ・待機揺れが競合しないようにする。",
                    ["レイヤごとの加算・上書きのルール", "20Hz 出力"],
                    ["合成ロジックの単体テスト"],
                ),
            ),
            (
                "モーションの追加（受付らしい所作）",
                ["type:feature", R, "area:expression", "needs-robot", "P2"],
                body(
                    "お辞儀・歓迎・見送りなど。",
                    ["welcome / bow / see_off / proud / thinking などを追加"],
                    ["実機で自然に見える（オーナー確認）"],
                ),
            ),
            (
                "フィラーの再生で体感遅延を短縮",
                ["type:feature", R, "area:expression", "P1"],
                body(
                    "2 秒を超えそうなときに沈黙を埋める。",
                    [
                        "サーバーから配布されたフィラーのキャッシュ",
                        "turn.start から最初の音声までの待ち時間に応じて再生",
                    ],
                    ["レイテンシのログで効果を確認"],
                    "フィラーと定型挨拶の事前合成",
                ),
            ),
        ],
    },
    # ------------------------------------------------------------------ M9
    {
        "milestone": "M9 運用と評価",
        "title": "[Epic] M9 運用と評価: 品質評価・メトリクス・長期運用",
        "desc": "研究室で毎日動かし続けられる状態にする。設計書 §5。",
        "tasks": [
            (
                "LLM 応答品質の評価ワークフロー（手動・夜間、LLM-as-judge）",
                ["type:ci", S, "area:agent", "P1"],
                body(
                    "プロンプト変更の回帰を確認する。API コストと非決定性があるので PR の CI には含めない。",
                    [
                        "評価シナリオ（親密度別の挨拶など）",
                        "workflow_dispatch で実行し、レポートを Artifact にする",
                    ],
                    ["手動実行でレポートが出る"],
                    "受付嬢ペルソナ",
                ),
            ),
            (
                "GPU マシンをセルフホストランナーにした結合テスト",
                ["type:ci", S, "needs-gpu", "P2"],
                body(
                    "本物の STT / TTS を使った結合テストを定期的に回す。",
                    [
                        "セルフホストランナーの登録手順（public リポジトリなのでフォーク PR では動かさない）",
                        "夜間ワークフロー",
                    ],
                    ["夜間に結合テストが走る"],
                    "サーバーのデプロイ",
                ),
            ),
            (
                "運用メトリクス（来訪数・挨拶数・会話ターン・レイテンシ）",
                ["type:feature", "area:ops", "P2"],
                body(
                    "研究室に来るモチベーションへの効果を観察する。",
                    ["ロボット側の日次集計", "サーバー側のレイテンシ集計", "簡易レポート"],
                    ["1 日分のレポートが出る"],
                    "来訪ログ, トレースとレイテンシ計測",
                ),
            ),
            (
                "1 週間の連続運用テスト",
                ["type:research", "area:ops", "needs-robot", "needs-gpu", "P1"],
                body(
                    "長時間の運用での安定性（メモリリーク・熱・API コスト）。",
                    ["連続運用", "見つかった問題を個別の Issue にする"],
                    ["1 週間止まらずに動く"],
                    "ロボットの systemd, サーバーのデプロイ",
                ),
            ),
        ],
    },
]
