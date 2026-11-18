# リーチ受付嬢 — システム設計書

> Reachy Mini Wireless を研究室の入口に置き、入室者に日常的に声をかけ、研究室に来るモチベーションを上げる「研究室の一員」エージェント。
> 本書は既存実装（Phase 0〜5: LangGraph + WS + SBV2 + YuNet 視線追従 + 顔ベース人物メモリ）を土台にした **目標アーキテクチャ** と **移行計画** を定める。

---

## 0. 要件と設計方針の対応

| # | 要件 | 設計上の答え | 主担当コンポーネント |
|---|---|---|---|
| R1 | 自然な音声対話（AIエージェント） | 文単位ストリーミング（STT→LLM→TTS）＋相槌/フィラーで体感遅延を隠す。LangGraph のステートグラフで対話を制御 | `agent/`, `expression/tts` |
| R2 | 対話内容に合わせた身体動作 | LLM が **文ごとに感情/意図タグ** を出し、Expression Planner がモーションへ変換。反射層（視線追従・待機揺れ・相槌うなずき）は LLM を通さない | `expression/` |
| R3 | カメラで顔を覚え状況を判断 | 端末上で YuNet（検出）＋SFace（埋め込み）＋多人数トラッカー。状況（人数・接近/通過・視線）は World Model に集約。重い場面理解だけ VLM | `perception/`, `world/` |
| R4 | 来訪者情報を蓄積し、頻度で話題を変える | SQLite に「人物・顔埋め込み・来訪ログ・事実・会話要約」。来訪統計から **親密度区分**（初対面/常連/久しぶり/毎日）を算出し、挨拶戦略とプロンプトを切替 | `memory/` |
| R5 | 研究室の一員として、話す番か・自分宛てかを判断 | **Floor Manager（ターンテイキング/宛先判定）** を独立モジュール化。名前呼び・発話中アイコンタクト・DoA と顔の対応・人数・直前のロボット発話・発話内容を特徴量に RESPOND / IGNORE / BACKCHANNEL を決定。**タイマー式の自発発話は禁止** | `social/` |
| R6 | ハーネスをしっかり設計 | 型付きイベントバス、プロバイダ抽象（STT/LLM/TTS/VLM）、設定・ログ・トレース、**セッション記録とリプレイ**、シナリオ評価、ウォッチドッグ、フォールバック | `core/`, `harness/` |
| R7 | 可能なら Reachy Mini 単体で処理 | 端末上で「知覚・判断・記憶・表現」を完結。CM4 で実時間処理が無理な STT/LLM/TTS は **クラウド API を差し替え可能なプロバイダ** として呼ぶ。自前 PC サーバーは任意（`edge+pc` プロファイル） | `adapters/`, デプロイプロファイル |

### 設計原則
1. **反射層と熟考層の分離** — 視線追従・相槌・待機モーションはローカルで即時（<100ms）。LLM は「何を話すか」だけを担う。
2. **判断はコード、表現は LLM** — 「話すべきか」「誰に向けた発話か」は検証可能なルール＋スコアで決める（テスト可能・説明可能）。LLM 判定は曖昧領域の補助に限定。
3. **純粋ロジックと I/O の分離** — 判定ロジックは副作用のない関数/クラスにし、ロボット SDK・クラウド API はアダプタ越しに注入。CI で実機なしに全ロジックをテストできる。
4. **すべてのイベントは記録でき、再生できる** — 実機で起きた事象を JSONL に記録し、オフラインで同じ入力列を流して回帰テストする。
5. **プライバシーはデフォルトで守る** — 顔埋め込み・会話記録は端末内のみ。忘却要求・保持期限を最初から設計に含める。

---

## 1. デプロイ構成

### 1.1 ハードウェア制約（Reachy Mini Wireless / Raspberry Pi CM4, Debian 13 aarch64）

| 処理 | CM4 上での実測/見込み | 判断 |
|---|---|---|
| VAD | XVF3800 内蔵（`get_DoA().is_speech`） | **端末** |
| DoA（音源方向） | XVF3800 内蔵 | **端末** |
| 顔検出 YuNet 320px | 10〜20 fps 見込み（Phase 3B で実機稼働済み） | **端末** |
| 顔埋め込み SFace | 1顔 数十 ms 見込み、同定は数 Hz で十分 | **端末**（要ベンチ: Issue） |
| 人物 DB / 来訪統計 | SQLite | **端末** |
| オーケストレーション（LangGraph） | Python のみ、軽量 | **端末** |
| STT Whisper base | 5s 音声に 20〜30s（実測） | ✗ → **クラウド** |
| LLM | — | **クラウド** |
| TTS SBV2 | CPU で実時間困難 | ✗ → **クラウド** または `edge+pc` |

### 1.2 デプロイプロファイル

```
profile = edge+cloud（目標・推奨）           profile = edge+pc（現行互換）
┌──────────── Reachy Mini (CM4) ───────────┐   ┌──── Reachy Mini ────┐   ┌──── PC ────┐
│ perception / world / social / memory      │   │ perception / social  │   │ STT/LLM    │
│ agent(LangGraph) / expression / harness   │   │ expression           │◄─►│ TTS(SBV2)  │
│        │ HTTPS (STT / LLM / TTS / VLM)    │   │                      │WS │ agent      │
└────────┼──────────────────────────────────┘   └──────────────────────┘   └────────────┘
         ▼
   クラウド API（OpenAI 等、プロバイダは差し替え可能）
```

- **edge+cloud**: 自前サーバー不要。Reachy の電源を入れれば systemd で起動。声は クラウド TTS（キャラクター性は声選択＋プロンプトで補う）。
- **edge+pc**: 学長モデル SBV2 の声を使いたい場合。PC は「TTS（と任意で STT）のリモートプロバイダ」に格下げし、**判断・記憶は常に端末側** に置く（PC が落ちてもクラウド TTS にフォールバック）。
- 切替は設定 `RUNTIME_PROFILE` とプロバイダ registry のみで行い、上位ロジックは変えない。

### 1.3 レイテンシ予算（発話終了 → ロボットの第一声）

| 区間 | 目標 |
|---|---|
| VAD 終端確定 | 0.4〜0.6 s |
| 宛先判定（ルール） | < 10 ms（LLM 補助時 < 400 ms） |
| 相槌うなずき（反射） | VAD 終端から < 150 ms で開始 |
| STT（クラウド） | 0.4〜1.0 s |
| LLM 最初の文 | 0.5〜1.0 s |
| TTS 最初のチャンク | 0.3〜0.6 s |
| **合計（体感）** | **≦ 2.5 s**、それを超える場合は短いフィラー（「えっとね」等）を先行再生 |

---

## 2. 論理アーキテクチャ

```
                       ┌──────────────────────── harness ────────────────────────┐
                       │ config · logging/trace · recorder/replay · watchdog · metrics │
                       └──────────────────────────────────────────────────────────┘
 ┌─────────── Perception ───────────┐      ┌────── World Model ──────┐
 │ AudioIn  : VAD区間 / DoA          │      │ people_present[track]   │
 │ Vision   : 検出→トラック→同定      │─ev─► │ speaking_track / floor  │
 │ GazeEst  : アイコンタクト推定       │      │ robot_state(speaking..) │
 │ EntryDet : 接近/入室/通過/退出      │      │ time / calendar         │
 └──────────────────────────────────┘      └──────────┬──────────────┘
                 │ events (typed, async bus)          │ snapshot
                 ▼                                    ▼
 ┌──────────── Social (判断) ─────────────────────────────────────┐
 │ FloorManager   : この発話は自分宛てか？ 今は自分の番か？        │
 │                  → RESPOND / BACKCHANNEL / IGNORE               │
 │ InitiativePolicy: 自分から話しかけるべきか？（入室・目が合った）  │
 │                  → GREET(strategy) / ENGAGE / NONE               │
 │ BargeIn        : ロボット発話中に話しかけられたら中断           │
 └───────────────┬────────────────────────────────────────────────┘
                 │ Intent(turn / proactive, target person, strategy)
                 ▼
 ┌──────────── Agent (LangGraph, 熟考層) ───────────────────────────┐
 │ build_context → plan/respond(LLM+tools) → guard → emit(文+感情)   │
 │ tools: memory(recall/remember/forget/set_name) · lab_info ·       │
 │        time · weather · express_motion · look_at(person)           │
 └───────────────┬──────────────────────────┬───────────────────────┘
                 │ sentences + emotion tags  │ memory writes
                 ▼                           ▼
 ┌──────── Expression ────────┐   ┌──────── Memory (SQLite) ────────┐
 │ TTS stream · filler        │   │ persons · face_embeddings        │
 │ ExpressionPlanner(感情→動作) │   │ visits · facts · episodes(要約)   │
 │ Reflex: gaze/idle/nod       │   │ familiarity 算出 · 忘却/保持期限   │
 │ MotionController(20Hz)      │   └──────────────────────────────────┘
 └─────────────────────────────┘
```

### 2.1 イベントバス（`core/events.py`）
`asyncio` ベースの pub/sub。全イベントは `pydantic` モデルで型付けし、`ts`（monotonic + wall）と `trace_id` を持つ。

| イベント | 発生元 | 主な購読者 |
|---|---|---|
| `FaceTracked(track_id, bbox, landmarks, frontal_metrics)` | Vision | World, Gaze |
| `PersonIdentified(track_id, person_id, confidence, is_new)` | Vision | World, Memory |
| `PersonEntered / PersonPassedBy / PersonLeft(track_id)` | EntryDet | Initiative, Memory(visits) |
| `EyeContactChanged(track_id, on, duration)` | GazeEst | Initiative, Floor |
| `SpeechStarted / SpeechEnded(doa, audio_ref)` | AudioIn | Floor, Reflex(相槌) |
| `UtteranceTranscribed(text, speaker_track?, doa)` | STT | Floor |
| `FloorDecision(decision, reasons, scores)` | Floor | Agent, Recorder |
| `ProactiveIntent(kind, person_id, strategy)` | Initiative | Agent |
| `RobotSpeechStarted / Ended(text)` / `BargeIn` | Expression | Floor, Initiative |

**全イベントは Recorder が JSONL に書き出す** → Replay ハーネスで再生可能（§5）。

### 2.2 World Model（`world/`）
イベントを畳み込んだ「いまの状況」のスナップショット。判断モジュールは World の **読み取り専用スナップショット** だけを見る（テスト時はスナップショットを直接組み立てる）。
- `people_present: dict[track_id, TrackState]`（person_id, 距離指標 face_frac, 正面度, アイコンタクト継続秒, 入室からの経過）
- `speaking_track`（DoA と顔の水平位置の対応で推定）
- `robot: {speaking, last_utterance, last_question_to: person_id, last_proactive_at}`
- `conversation: {active_with: person_id | None, turns, last_turn_at}`

---

## 3. 社会的判断（Social）— 本システムの肝

### 3.1 FloorManager（宛先判定・ターンテイキング）
入力: `UtteranceTranscribed` ＋ World スナップショット。出力: `RESPOND | BACKCHANNEL | IGNORE` と理由。

**特徴量（すべてログに残す）**

| 特徴 | 例 | 重み方針 |
|---|---|---|
| 名前呼び | 「リーチ」「りーち」「reachy」（STT 揺れ辞書） | 強い正（単独で RESPOND） |
| 発話中アイコンタクト | 発話区間のうち正面を向いていた割合 | 強い正 |
| 話者=注視者 | DoA 方向の顔がロボットを見ているか | 正 |
| 人数 | 1人なら独り言/自分宛ての事前確率↑、複数人なら↓ | 事前確率 |
| 会話継続 | 直近 N 秒以内にロボットがこの人に質問した | 強い正 |
| 人同士の会話 | 複数の顔が互いの方向を向いている/交互に話している | 強い負 |
| 発話内容 | 二人称・ロボット話題・挨拶 vs 第三者の名前・業務会話 | 弱〜中（曖昧時のみ LLM 分類） |

**決定ロジック**: ルールスコア → しきい値で 3 値。スコアが曖昧帯のときだけ軽量 LLM 分類（「この発話はロボット宛てか」JSON 出力、タイムアウト 400ms、失敗時は IGNORE 側に倒す）。
**原則: 迷ったら黙る**（割り込みの不快 > 無視の不快）。ただし BACKCHANNEL（うなずき等の非言語）は「聞いているよ」を低コストで示せるので曖昧帯で使う。

### 3.2 InitiativePolicy（自発発話）
**タイマー式は禁止**（既存方針を継承）。自発発話してよい契機:
1. **入室検知** — 顔が一定以上の大きさに近づき、入口方向から入ってきた（通過ではない）。
2. **持続アイコンタクト** — ENGAGE_SEC（既定 2.0s）以上こちらを見ている。
3. **（将来）困っていそう** — VLM による判定。

抑制条件: ロボット発話中 / 人同士が会話中 / 同一人物への自発発話から MIN_GAP 未満 / 同一人物への入室挨拶は **1日1回**（退出→再入室は短い一言のみ）/ 相手が作業中（よそ見・PC 注視）。

### 3.3 親密度区分と挨拶戦略（R4）
来訪ログから算出（`memory/familiarity.py`、純粋関数）:

| 区分 | 判定（初期値、設定可能） | 挨拶戦略の例 |
|---|---|---|
| `first_time` | 来訪記録なし | 自己紹介＋名前を尋ねる |
| `new` | 来訪 2〜4 回 | 名前で呼ぶ＋前回の話題に軽く触れる |
| `daily` | 直近 7 日中 5 日以上 | 短く軽快に。連続日数に触れて称える（「5日連続だね！」） |
| `regular` | 直近 30 日に 4 回以上 | 名前＋覚えている事実を 1 つ |
| `long_absence` | 最終来訪から 14 日以上 | 「久しぶり！」＋何してたか尋ねる、来てくれたことを喜ぶ |
| `returning_today` | 本日 2 回目以降 | 「おかえり」程度の一言 |

時間帯（朝/昼/夕/深夜）・曜日（月曜/金曜）・最初の来訪者か、なども戦略に加える。
**モチベーション設計**: 来訪を「承認」する（連続日数・早い到着・久々の来訪を喜ぶ）。説教・ノルマ的な発言（「最近来てないね」の責め口調）はプロンプトとガードで禁止。

---

## 4. エージェント（LangGraph）

### 4.1 グラフ
`create_react_agent` から **明示的な StateGraph** に移行する。

```
START → build_context → respond ⇄ tools → guard → emit → END
```

- **build_context**: Intent（応答 or 自発）、相手プロフィール（名前・親密度・事実・前回要約）、状況（人数・時間帯）を **構造化コンテキスト** としてシステムメッセージに注入（現在の「（相手メモ：…）」テキスト前置きを置換）。
- **respond**: LLM。出力は文ごとに `<emo:happy>` 等の感情タグを付けられる軽量フォーマット（ストリーミングと両立）。
- **tools**: ToolNode。
- **guard**: 長さ（音声なので 1〜3 文）、禁止事項（容姿評価・機微情報の詮索・責め口調）、読み上げ不能記号の正規化。
- **emit**: 文＋感情タグ → Expression へ。

**State**: `messages`, `intent`, `person`, `situation`, `emotion_plan`。
**会話スレッド**: thread_id = `person_id:date`（人ごと・日ごと）。長期の記憶は episodes（要約）として Memory に退避し、スレッドは短く保つ。チェックポイントは `SqliteSaver`（端末内）。

### 4.2 ツール
| ツール | 用途 |
|---|---|
| `set_person_name`, `remember_fact`, `forget_me`, `recall_person` | 人物メモリ（既存＋忘却・想起） |
| `lab_presence` | 「今日は誰が来てる？」「〇〇さん今日来た？」（来訪ログを検索、本人同意済みの範囲のみ） |
| `express_motion`, `look_at_person` | 身体表現（既存） |
| `get_current_time`, `get_weather` | 既存 |
| （将来）`lab_calendar` | ゼミ・締切などの予定 |

ツールは ContextVar で「いまの相手」「モーションシンク」を受け取る既存方式を踏襲。

### 4.3 プロンプト/ペルソナ
「研究室の受付も兼ねるメンバーのリーチ」。入口にいて、来た人を覚えていて、来てくれると嬉しい。研究室メンバー同士の会話には割り込まない。プロンプトは `agent/prompts/` にファイル分割（persona / speaking_style / greeting_strategies / safety）し、評価シナリオで回帰確認する。

---

## 5. ハーネス（R6）

| 要素 | 内容 |
|---|---|
| **設定** | `pydantic-settings`。`.env` + 環境変数。しきい値（EC_*, ENGAGE_SEC, MIN_GAP, 親密度境界…）は全て設定化し、起動時にダンプ |
| **プロバイダ registry** | `STTProvider` / `LLMProvider` / `TTSProvider` / `VLMProvider` の Protocol。`openai`, `pc_remote`, `fake`（テスト用）実装。プロファイルで選択 |
| **ロボット抽象** | `RobotIO` Protocol（mic/DoA/camera/speaker/motors）。`ReachyRobotIO`（SDK）と `FakeRobotIO`（録画ファイル再生・CI 用） |
| **構造化ログ/トレース** | JSON ログ + `trace_id` で 1 ターンを追跡。区間レイテンシを計測。LangSmith は任意で有効化 |
| **Recorder / Replay** | 全イベントと（任意で）音声・低解像度フレームを `sessions/YYYYMMDD/*.jsonl` に記録。`replay` コマンドで判断層（Floor/Initiative/Agent）に同じ入力を流し、判定の差分を出す |
| **シナリオ評価** | YAML で状況と期待判定を記述（例: 2人が会話中に「リーチって可愛いよね」→ IGNORE か BACKCHANNEL）。判定層はルールなので CI で決定的に検証。LLM 出力の質は別ワークフロー（手動/夜間）で LLM-as-judge |
| **ウォッチドッグ** | 各ループの heartbeat、クラウド API タイムアウト・リトライ・サーキットブレーカ。失敗時は定型の短い返答＋モーションで沈黙を避ける |
| **運用** | systemd ユニット、ヘルスエンドポイント、ログローテーション、デプロイスクリプト |
| **メトリクス** | 1日の来訪者数・挨拶数・会話ターン・誤応答（人手ラベル）・平均レイテンシ。将来ダッシュボード化 |

---

## 6. 記憶（Memory, SQLite）

```sql
persons(id, name, created_at, consent_state, notes)
face_embeddings(id, person_id, vec BLOB, quality, created_at)   -- 上限 N 本/人
visits(id, person_id, entered_at, left_at, greeted, source)     -- 入室検知単位
facts(id, person_id, text, created_at, last_used_at, source_turn)
episodes(id, person_id, date, summary, created_at)              -- 会話要約
```

- 同定: コサイン類似（SFace 閾値 0.363 を初期値、実機で再調整）。同一トラック内は多数決で安定化。
- 来訪の確定: 同一人物のトラックが入口ゾーンで消え、T 分以上不在→再出現で新 visit。
- 要約: 会話終了（相手退出 or 一定時間無発話）時に LLM で 1〜2 文要約し episodes へ。
- **プライバシー**: 顔画像は保存しない（埋め込みのみ）。`forget_me` で人物・埋め込み・事実・来訪を削除。未命名の人物は N 日で自動削除。研究室内での掲示（「カメラで顔を覚えます」）を運用ルールに含める。
- 移行: 既存 `data/people.json` → SQLite のマイグレーションスクリプト。

---

## 7. 表現（Expression）

- **TTS ストリーミング**: 文単位（既存）。クラウド TTS は音声チャンクのストリーム受信に対応。
- **ExpressionPlanner**: 感情タグ（happy / surprised / thinking / sad / shy / proud）→ モーションプリミティブ（既存 `motion.py` のライブラリを拡張）。文の再生開始に合わせて発火。
- **反射層**: 視線追従（既存 GazeTracker）、待機揺れ・話中モーション（既存 MotionController）、**聞いている間のうなずき（新規）**、話者への振り向き（DoA）。
- **優先度合成**: 反射（視線）＋ジェスチャ（感情）＋ベース（idle）を加算/上書きルールで合成し 20Hz で出力。

---

## 8. リポジトリ構成（目標）

```
reachy-mini-chat/
├── reachy_agent/             # 端末で動く本体（新パッケージ）
│   ├── core/                 # events, bus, config, clock, logging  ← 純粋
│   ├── perception/           # audio_in, vision(track/identify), gaze, entry
│   ├── world/                # WorldModel（イベント畳み込み）         ← 純粋
│   ├── social/               # floor, initiative, bargein              ← 純粋
│   ├── memory/               # store(SQLite), familiarity, summarizer
│   ├── agent/                # graph, state, prompts/, tools/
│   ├── expression/           # tts_stream, planner, motion, reflex
│   ├── adapters/             # robot_io(reachy/fake), providers(openai/pc_remote/fake)
│   ├── harness/              # recorder, replay, scenarios, watchdog, metrics
│   └── app.py                # エントリポイント（プロファイル選択して起動）
├── pc_server/                # edge+pc プロファイル用リモートプロバイダ（縮小して残す）
├── robot/                    # 既存の診断スクリプト群（段階的に reachy_agent へ移行）
├── tests/
│   ├── unit/                 # 純粋ロジック
│   ├── scenarios/            # 判定シナリオ YAML + runner
│   └── fixtures/             # 記録済みセッション・画像
├── deploy/                   # systemd unit, deploy.sh
└── docs/                     # 本書・ADR
```

既存コードは段階的に移す（一括書き換えはしない）。各 Issue は「動く状態を保ったまま 1 歩進む」単位。

---

## 9. CI / 開発フロー

- **GitHub Actions（PR と main への push で実行）**
  - `lint`: ruff check / ruff format --check
  - `typecheck`: pyright（`reachy_agent/core,world,social,memory` は strict から開始）
  - `test`: pytest（unit + scenarios）＋カバレッジ。重い ML 依存（torch/whisper/SBV2）は CI ではインストールしない（依存グループ分割）
  - `lock`: `uv lock --check`（ロックファイルの整合）
  - `robot-compat`: 端末向けコードが Python 3.12 / aarch64 相当で import 可能か（SDK はスタブ）
- **ブランチ運用**: `main` 保護（PR 必須・CI 必須・レビュー 1 承認）。ブランチ名 `feat/<issue#>-<slug>`、PR 本文に `Closes #<issue>`。
- **開発サイクル**: Claude が Issue を取る → ブランチ → 実装＋テスト → PR → オーナーがレビュー＆マージ → 次の Issue。
- **実機検証**: CI では不可能なので、PR テンプレートに「実機確認手順/結果」欄を設け、必要な PR はオーナーが実機で確認。

---

## 10. マイルストーンと Issue 計画

| マイルストーン | 内容 | 完了条件 |
|---|---|---|
| **M0 基盤整備** | 現行コードの main 取り込み、CI、依存分割、テスト基盤、PR/Issue テンプレート | main が CI グリーン、保護ルール有効 |
| **M1 ハーネス中核** | 新パッケージ骨格、設定、イベントバス、プロバイダ/RobotIO 抽象、Recorder/Replay、構造化ログ | Fake 構成で E2E（音声ファイル→応答）が CI で通る |
| **M2 端末単体化** | クラウド STT/LLM/TTS プロバイダ、端末上ランタイム、systemd、CM4 ベンチ | PC なしで実機会話が成立 |
| **M3 知覚** | 多人数トラッカー、端末上顔同定、視線推定移植、DoA-顔対応、入室検知 | 入室/通過/退出イベントが実機で妥当 |
| **M4 記憶** | SQLite 化・移行、来訪ログ、親密度算出、会話要約、忘却 | 親密度区分ごとに挨拶が変わる |
| **M5 社会的判断** | FloorManager、InitiativePolicy、BargeIn、相槌、シナリオ評価 | シナリオ評価が CI で全件パス |
| **M6 エージェント刷新** | StateGraph 化、構造化コンテキスト、ペルソナ、新ツール、ガード | 評価シナリオで品質確認 |
| **M7 表現** | 感情タグ→モーション、うなずき、モーション合成、フィラー | 実機で発話と動作が同期 |
| **M8 運用・評価** | LLM 品質評価ワークフロー、メトリクス、デプロイ自動化 | 1 週間の連続運用 |

Issue はマイルストーンごとに Epic（親 Issue）を立て、作業単位を Sub-issue としてぶら下げる。

---

## 11. 未決事項（オーナー判断が必要）

1. **声**: edge+cloud ではクラウド TTS の声になる。学長 SBV2 の声を残すなら edge+pc を併用。→ 当面は両対応（プロバイダ切替）で進め、既定は edge+cloud を推奨。
2. **クラウドプロバイダ**: 現行 OpenAI を継続（STT: gpt-4o-mini-transcribe 系、LLM: gpt-4o-mini 系、TTS: gpt-4o-mini-tts 系）を前提。変更する場合はプロバイダ追加で対応。
3. **研究室メンバー情報**: 名前・所属を事前登録するか、会話で覚えさせるか（初期は会話で覚える）。
4. **掲示と同意**: 顔を覚えることの掲示・オプトアウト方法の運用。
