# リーチ受付嬢 — システム設計書

> Reachy Mini Wireless を研究室の入口に置き、入室者に日常的に声をかけ、研究室に来るモチベーションを上げる「研究室の一員」エージェント。
> 本書は既存実装（Phase 0〜5: LangGraph + WS + SBV2 + YuNet 視線追従 + 顔ベース人物メモリ）を土台にした **目標アーキテクチャ** と **移行計画** を定める。
>
> **改訂 2 (2026-11-18)**: 構成を「ロボット＝フロントエンド / 研究室 GPU サーバー＝生成バックエンド」に変更。

---

## 0. 要件と設計の対応

| # | 要件 | 設計上の答え | 担当 |
|---|---|---|---|
| R1 | 自然な音声対話（AI エージェント） | サーバーで STT→LLM(LangGraph)→TTS を文単位ストリーミング。ロボットは受け取った文から順に再生し、相槌・フィラーで体感遅延を隠す | server / robot |
| R2 | 対話内容に合わせた身体動作 | サーバーは文ごとに **感情タグ** を返し、ロボットの Expression Planner がモーションに変換する。視線追従・待機揺れ・うなずきなどの反射はロボットだけで完結させる | robot |
| R3 | カメラで顔を覚え、状況を判断 | ロボット上で YuNet（検出）、SFace（埋め込み）、多人数トラッカー、視線推定、入室検知を動かす。場面理解（VLM）が必要なときだけ縮小フレームをサーバーへ送る | robot |
| R4 | 来訪者情報を蓄積し、頻度で話題を変える | **人物の記憶はロボット側**（SQLite: 人物・顔埋め込み・来訪・事実・会話要約）。来訪統計から親密度を算出し、毎ターンのリクエストにプロフィールとして添付する | robot |
| R5 | 話す番か・自分宛てかを判断 | **判断はロボット側**。サーバーから返る文字起こしと、ロボットが持つ視線・話者・人数・会話状態を合わせて RESPOND / BACKCHANNEL / IGNORE を決める。タイマー式の自発発話は禁止 | robot |
| R6 | ハーネスをしっかり設計 | 版管理された **共有プロトコル**、両側のプロバイダ抽象と Fake、記録と再生、シナリオ評価、契約テスト、トレース、ウォッチドッグ、フォールバック | protocol / 両側 |
| R7 | 処理の分担 | ロボットは知覚・判断・記憶・API 呼び出し・表現の「フロントエンド」。重い推論（STT/LLM/TTS）と会話履歴は自作の **研究室 GPU サーバー** に任せる | 全体 |

### 設計原則
1. **ロボットが主、サーバーは従** — 「いつ・誰に・話すか」はロボットが決め、サーバーは「何を・どんな声で話すか」を生成する。サーバーから自発的に話し始めることはない。
2. **反射層と熟考層の分離** — 視線追従・うなずき・待機モーションはロボット内で 100ms 未満で反応する。サーバーが落ちていても止まらない。
3. **判断はコード、表現は LLM** — 宛先判定や自発発話の判断はテストできるルールとスコアで行い、LLM は曖昧なときの補助に限る。
4. **プロトコルが契約** — ロボットとサーバーの間のメッセージは `protocol/` パッケージで型と版を定義し、両側の CI で契約テストを回す。
5. **すべて記録でき、再生できる** — 両側のイベントを JSONL に記録し、オフラインで回帰テストする。
6. **プライバシーはデフォルトで守る** — 顔埋め込みはロボット内だけに置く。サーバーへは不透明な person_id とプロフィール文だけを送る。忘却要求は両側に伝える。

---

## 1. システム構成

```
┌──────────────── Reachy Mini (CM4) = フロントエンド ────────────────┐        ┌──────── 研究室 GPU サーバー = 生成バックエンド ───────┐
│                                                                    │        │                                                      │
│  Perception: VAD/DoA(XVF3800) · YuNet · SFace · Tracker · Gaze     │        │  STT  : faster-whisper (GPU)                         │
│      │                                                             │        │  Agent: LangGraph (LLM はクラウド API / 将来ローカル) │
│  World Model ─► Social: FloorManager / InitiativePolicy / BargeIn  │  WS    │  TTS  : Style-Bert-VITS2 (GPU, 学長モデル)           │
│      │                         │                                   │◄──────►│  History: 会話スレッド（person_id × 日付）           │
│  Memory (SQLite): 人物・顔・来訪・事実・要約 ─► Profile 添付        │ 1接続  │  Server tools: 時刻・天気・(予定)                    │
│      │                         │                                   │        │  Harness: registry · trace · eval · health           │
│  Server Client (再接続・取消・タイムアウト・フォールバック)         │        │                                                      │
│      │                                                             │        └──────────────────────────────────────────────────────┘
│  Expression: 音声再生 · ExpressionPlanner · Reflex · Motion 20Hz    │
└────────────────────────────────────────────────────────────────────┘
```

### 1.1 責務の分担

| 処理 | ロボット | サーバー | 理由 |
|---|---|---|---|
| VAD・DoA | ● | | ハードウェア内蔵 |
| 顔検出・追跡・視線・入室検知 | ● | | 反射に必要。映像を外へ出さない |
| 顔同定（SFace 埋め込み・照合） | ● | | 生体情報をロボット内に閉じる |
| 人物の記憶・来訪ログ・親密度 | ● | | ユーザー方針（ロボット＝記憶を持つフロントエンド） |
| 宛先判定・自発発話の判断 | ● | | 視線・人数などの状況はロボットだけが知っている |
| STT | | ● | CM4 では実時間に間に合わない |
| 応答生成（LLM・ツールループ） | | ● | |
| 会話履歴（短中期） | | ● | ユーザー方針。スレッドは person_id × 日付 |
| 会話要約 | | ●（生成） | 生成した要約はロボットに返して保存する |
| TTS（SBV2） | | ● | GPU で高速に合成する |
| モーション・音声再生 | ● | | |
| 研究室の情報を引くツール（来訪状況など） | ●（実行） | ●（呼出） | データはロボットにある → サーバーから RPC で呼ぶ |

### 1.2 ネットワークと可用性
- ロボットとサーバーは研究室 LAN 上の WebSocket で 1 本の常時接続にする。認証は事前共有トークン。学内ネットワークをまたぐ場合は TLS（wss）にする。
- **サーバーが落ちているとき**: ロボットは知覚・反射・記憶を続ける。入室者には、事前に合成してロボットに置いた短い挨拶音声とモーションで反応し、会話はできないことを示す。サーバーが復帰したら自動で再接続する。
- サーバーは 1 台のロボットを前提にするが、`robot_id` でスレッドを分けておき、複数台にも広げられるようにする。

### 1.3 レイテンシ予算（発話終了 → 第一声）

| 区間 | 目標 |
|---|---|
| VAD 終端の確定（ロボット） | 0.4〜0.6 s |
| 相槌うなずき（ロボット反射） | VAD 終端から 150 ms 未満 |
| 音声送信（LAN） | < 50 ms |
| STT（GPU, faster-whisper） | 0.2〜0.5 s |
| 宛先判定の往復（transcript → decision） | < 30 ms（ルールのみ） |
| LLM の最初の文 | 0.5〜1.0 s |
| TTS 最初の文（GPU, SBV2） | 0.1〜0.3 s |
| **合計（体感）** | **2.0 s 以下**。超えそうなときはロボット側のフィラーを先に再生する |

---

## 2. プロトコル（`protocol/`）— ロボットとサーバーの契約

pydantic のモデルで定義し、両側が同じパッケージを import する。`protocol_version` は semver で管理し、メジャー版が違えば接続時に拒否する。

### 2.1 フレーミング
- **text フレーム**: JSON の制御メッセージ。`{"type": ..., "id": ..., "ts": ..., ...}`
- **binary フレーム**: `[4 バイトのヘッダ長][JSON ヘッダ][ペイロード]`。ヘッダには `kind`（`audio_in` / `audio_out` / `frame`）、`ref`（utt_id か turn_id）、`seq`、`sr`、`format` を入れる。

### 2.2 1 ターンの流れ（ユーザーの発話に応答する場合）

```
robot                                              server
  │── utterance.begin {utt_id, doa, person_hint} ──►│
  │══ audio_in (binary, 発話中にストリーミング) ════►│  STT を逐次処理
  │── utterance.end {utt_id, features} ───────────►│
  │◄── transcript {utt_id, text, confidence} ──────│
  │  FloorManager が判定（文字起こし＋視線・人数など）│
  │── floor.decision {utt_id, decision, ...} ─────►│
  │    decision=respond の場合: context を添付     │
  │◄── turn.start {turn_id} ───────────────────────│  LangGraph 実行
  │◄── sentence {turn_id, idx, text, emotion} ─────│
  │◄══ audio_out (binary, turn_id, idx) ═══════════│  SBV2 合成
  │◄── memory.op {op_id, person_id, op, value} ────│  LLM のツール呼び出し
  │── memory.ack {op_id, applied|rejected} ───────►│
  │◄── turn.end {turn_id, status} ─────────────────│
```

- `decision=ignore` のとき、サーバーは応答しない。文字起こしは「場の文脈」として **短期バッファにだけ** 残す（既定 10 分、永続化しない、設定で無効化できる）。
- `decision=backchannel` のとき、サーバーは生成しない。ロボットがうなずきなどの非言語反応だけを返す。

### 2.3 メッセージ一覧

| 方向 | type | 内容 |
|---|---|---|
| R→S | `hello` | protocol_version, robot_id, token, capabilities |
| S→R | `welcome` | server_version, models（STT/LLM/TTS の名前） |
| R→S | `utterance.begin` / `utterance.end` | utt_id, DoA, 発話区間の特徴（視線率・人数・話者トラック） |
| S→R | `transcript` | utt_id, text, lang, confidence, 区間ごとの時刻 |
| R→S | `floor.decision` | utt_id, decision, reasons, `context` |
| R→S | `proactive.request` | intent_id, kind（greet / engage / see_off）, strategy, `context` |
| R→S | `context`（埋め込み） | person: {id, name, familiarity, streak, days_absent, facts[], last_summary}, situation: {people_count, time_of_day, weekday, others_present[]}, frame_ref? |
| S→R | `turn.start` / `turn.end` | turn_id, ref（utt_id か intent_id）, status（ok / cancelled / error） |
| S→R | `sentence` | turn_id, idx, text, emotion |
| S→R | `motion` | turn_id, name（LLM が明示的に指定したジェスチャ） |
| S→R | `memory.op` | op_id, person_id, op（set_name / add_fact / add_episode / forget）, value |
| R→S | `memory.ack` | op_id, result, reason |
| S→R | `tool.call` | call_id, name（lab_presence など、ロボット側で実行するツール）, args |
| R→S | `tool.result` | call_id, result |
| R→S | `turn.cancel` | turn_id（割り込み時。サーバーは生成と合成を止める） |
| R→S | `person.forget` | person_id（サーバー側の会話履歴も削除する） |
| R→S | `session.close` | person_id（相手の退出。サーバーが要約を作り、`memory.op add_episode` で返す） |
| 双方向 | `ping` / `pong`, `error` | 死活監視、エラーコード |

### 2.4 記憶の書き込み（memory.op）
- 人物の記憶の **正はロボット側**。サーバーの LLM が「名前を覚える」「事実を覚える」と判断したら、ツールが `memory.op` を発行する。
- ロボットは検証してから保存する（名前らしさ、長さ、重複、機微情報のブロックリスト）。結果は `memory.ack` で返し、却下されたらサーバーはその事実を以後の文脈に使わない。
- サーバーの会話履歴（スレッド）は person_id 単位で持つが、名前や事実の正本はロボットが毎ターン送る `context.person` とする。

---

## 3. ロボット（フロントエンド, `robot_app/`）

### 3.1 内部構成

| モジュール | 役割 |
|---|---|
| `core/` | 設定（pydantic-settings）、型付きイベントバス（asyncio）、時計、構造化ログ |
| `io/` | `RobotIO` Protocol（マイク・DoA・カメラ・スピーカー・首・アンテナ）。`ReachyRobotIO`（SDK）と `FakeRobotIO`（録画を再生） |
| `perception/` | 音声区間、顔トラッカー、SFace 同定（トラック内の多数決）、視線推定、DoA と顔の対応付け、入室検知 |
| `world/` | イベントを畳み込んだ状況スナップショット（在室者・話者・ロボット状態・会話状態） |
| `social/` | FloorManager、InitiativePolicy、BargeIn |
| `memory/` | SQLite ストア、来訪ログ、親密度、プロフィール構築、memory.op の検証と適用 |
| `client/` | サーバー接続（再接続・ハートビート・タイムアウト・取消・ストリーム受信） |
| `expression/` | 音声再生キュー、ExpressionPlanner（感情→モーション）、反射（視線・うなずき・待機）、モーション合成 20Hz、フォールバック音声 |
| `harness/` | Recorder / Replay、シナリオ評価ランナー、ウォッチドッグ |
| `app.py` | 起動。systemd で常駐させる |

### 3.2 イベント（ロボット内部）

| イベント | 発生元 | 主な購読者 |
|---|---|---|
| `FaceTracked`, `PersonIdentified`, `PersonEntered / PassedBy / Left` | perception | world, memory, social |
| `EyeContactChanged` | perception | social |
| `SpeechStarted / SpeechEnded` | perception | client（音声送信）, expression（相槌）, social |
| `TranscriptReceived` | client | social（FloorManager） |
| `FloorDecided`, `ProactiveDecided` | social | client, recorder |
| `SentenceReceived`, `AudioChunkReceived`, `TurnEnded` | client | expression |
| `RobotSpeechStarted / Ended`, `BargeIn` | expression | social, client |
| `ServerConnectionChanged` | client | expression（フォールバック）, social |

### 3.3 FloorManager（宛先判定）
`TranscriptReceived` と World のスナップショットから、RESPOND / BACKCHANNEL / IGNORE と理由を出す。

| 特徴 | 例 | 重み |
|---|---|---|
| 名前呼び | 「リーチ」「りーち」「reachy」（STT の表記ゆれ辞書） | 強い正（それだけで RESPOND） |
| 発話中のアイコンタクト | 発話区間のうち正面を向いていた割合 | 強い正 |
| 話者＝注視者 | DoA の方向の顔がロボットを見ているか | 正 |
| 人数 | 1 人なら自分宛ての事前確率を上げ、複数人なら下げる | 事前確率 |
| 会話の継続 | 直近 N 秒以内にロボットがこの人に質問した | 強い正 |
| 人同士の会話 | 複数の顔が互いを向いている／交互に話している | 強い負 |
| 発話内容 | 二人称・挨拶・ロボットの話題か、第三者の名前・業務の会話か | 弱〜中（曖昧なときだけサーバーの分類 API を使う） |

**原則: 迷ったら黙る。** 割り込まれる不快さは、無視される不快さより大きい。曖昧なときは BACKCHANNEL（うなずき）で「聞いているよ」と示す。

### 3.4 InitiativePolicy（自発発話）
タイマー式は禁止（既存方針を継承）。話しかけてよい契機:
1. **入室検知** — 入口の方向から一定の大きさまで近づいた（通り過ぎではない）。
2. **持続アイコンタクト** — ENGAGE_SEC（既定 2.0 s）以上こちらを見ている。
3. **見送り** — 帰り際（退出の方向へ動いた）に短く声をかける。
4. （将来）困っていそうな様子 — サーバーの VLM で判定する。

抑制する条件: ロボットが発話中 / 人同士が会話中 / 同じ人への自発発話から MIN_GAP 未満 / 入室挨拶は同じ人に **1 日 1 回**（再入室は短い一言だけ）/ 作業中（よそ見・PC を注視している）。

### 3.5 親密度区分と挨拶の戦略

| 区分 | 判定（初期値、設定で変更可） | 挨拶の例 |
|---|---|---|
| `first_time` | 来訪記録なし | 自己紹介して名前を尋ねる |
| `new` | 来訪 2〜4 回 | 名前で呼び、前回の話題に軽く触れる |
| `daily` | 直近 7 日のうち 5 日以上 | 短く軽快に。連続日数を称える（「5 日連続だね！」） |
| `regular` | 直近 30 日に 4 回以上 | 名前＋覚えている事実を 1 つ |
| `long_absence` | 最終来訪から 14 日以上 | 「久しぶり！」と喜び、何をしていたか尋ねる |
| `returning_today` | 今日 2 回目以降 | 「おかえり」程度の一言 |

時間帯（朝・昼・夕・深夜）、曜日（月曜・金曜）、今日最初の来訪者か、なども `context.situation` に入れる。
**モチベーション設計**: 来訪を承認する（連続日数・早い到着・久しぶりの来訪を喜ぶ）。説教やノルマ的な言い方（「最近来てないね」という責める口調）はプロンプトとガードで禁止する。

### 3.6 記憶（SQLite, ロボット内）

```sql
persons(id, name, created_at, consent_state)
face_embeddings(id, person_id, vec BLOB, quality, created_at)   -- 1 人あたり上限 N 本
visits(id, person_id, entered_at, left_at, greeted)
facts(id, person_id, text, created_at, last_used_at, source)    -- source = memory.op の op_id
episodes(id, person_id, date, summary)                           -- サーバーが要約を生成して返す
```
- 顔画像は保存しない（埋め込みだけ）。未命名の人物は N 日で自動削除する。
- `forget_me` を受けたら、ロボットの全データを削除し、`person.forget` でサーバーの履歴も消す。
- 既存の `data/people.json` は移行スクリプトで取り込む。

---

## 4. サーバー（生成バックエンド, `server/`、研究室 GPU マシン）

### 4.1 内部構成

| モジュール | 役割 |
|---|---|
| `api/` | FastAPI。WS `/v1/session`、`/healthz`、`/v1/classify`（宛先判定の補助）、`/metrics` |
| `session/` | 接続ごとのセッション管理。ターンの状態機械（transcribing → waiting_decision → generating → done / cancelled） |
| `providers/` | `STTProvider`（faster-whisper GPU / OpenAI / Fake）、`LLMProvider`（OpenAI / 将来ローカル / Fake）、`TTSProvider`（SBV2 / OpenAI / Fake）。registry で選ぶ |
| `agent/` | LangGraph の StateGraph、プロンプト、ツール |
| `history/` | 会話スレッドの保存（SqliteSaver、thread_id = `robot_id:person_id:date`）、場の文脈の短期バッファ、要約の生成 |
| `harness/` | トレース（ターン単位のレイテンシ内訳）、記録、評価ランナー |

### 4.2 エージェント（LangGraph）

```
START → build_context → respond ⇄ tools → guard → emit → END
```
- **build_context**: ロボットから届いた `context`（相手のプロフィール・親密度・状況・自発発話の意図）を構造化してシステムメッセージに入れる。今の「（相手メモ：…）」をユーザー発話の前に付ける方式はやめる。
- **respond**: LLM。文ごとに `<emo:happy>` などの感情タグを付けた軽量な形式で出力する（ストリーミングと両立させる）。
- **tools**:
  - サーバーで実行: `get_current_time`, `get_weather`, （将来）`lab_calendar`
  - ロボットへ発行: `set_person_name`, `remember_fact`, `forget_me` → `memory.op`、`express_motion`, `look_at_person` → `motion`
  - ロボットへ RPC: `lab_presence`（「今日誰が来てる？」）, `recall_person` → `tool.call` / `tool.result`（タイムアウト 1 s）
- **guard**: 長さ（音声なので 1〜3 文）、禁止事項（容姿の評価・機微情報の詮索・責める口調）、読み上げられない記号の正規化。
- **emit**: 文と感情タグを TTS に流し、`sentence` と `audio_out` を送る。

### 4.3 TTS とストリーミング
- 文が完成するたびに SBV2 で合成して送る（既存の文分割ロジックを流用）。合成の待ち行列は 1 セッション 1 本にし、`turn.cancel` を受けたらすぐ破棄する。
- フィラー（「えっとね」など）と、サーバー障害時用の定型挨拶は、サーバーで事前に合成し、起動時にロボットへ配布してキャッシュさせる。

### 4.4 運用
- systemd（または Docker Compose）で常駐させる。GPU メモリは STT・TTS のモデルを常駐させて確保する。
- `/healthz` でモデルのロード状態を返す。ターンごとの区間レイテンシをログとメトリクスに出す。
- LLM のキーなどの秘密情報はサーバーだけに置き、ロボットには置かない。

---

## 5. ハーネス（R6）

| 要素 | ロボット | サーバー |
|---|---|---|
| 設定 | pydantic-settings（しきい値をすべて設定化し、起動時にダンプ） | 同じ（モデル・プロバイダの選択） |
| 抽象と Fake | `RobotIO` / `FakeRobotIO`、`ServerClient` / `FakeServer` | `STT/LLM/TTSProvider` と Fake |
| 契約テスト | `protocol/` のスキーマと順序規約（`ConversationChecker`）で送受信を検証。相手役は `FakeServer` | 同じ。**`FakeServer` と本物のサーバーが同じシナリオ（`reachy_protocol.testing`）を通るか** を CI で確認する |
| 記録と再生 | 全イベント（＋任意で音声・縮小フレーム）を JSONL に記録。判断層に再投入して判定の差分を出す | ターンの入出力と区間時間を記録 |
| シナリオ評価 | 宛先判定・自発発話の YAML シナリオを CI で決定的に検証する | LLM の応答品質は別ワークフロー（手動・夜間、LLM-as-judge） |
| 障害対策 | ウォッチドッグ、サーバー断のフォールバック | タイムアウト、サーキットブレーカ、プロバイダのフォールバック |
| 観測 | 来訪数・挨拶数・誤応答（人手ラベル） | レイテンシ内訳、トークン量、エラー率 |

---

## 6. リポジトリ構成（uv workspace）

```
reachy-mini-chat/
├── pyproject.toml            # workspace ルート（dev ツールのみ）
├── protocol/                 # reachy_protocol: メッセージ定義・版・フレーミング・順序規約（依存は pydantic のみ）
│                             #   reachy_protocol.testing: FakeServer / FakeRobotClient / 汎用の契約シナリオ
├── robot_app/                # reachy_front: ロボット上で動くフロントエンド
├── server/                   # reachy_server: GPU サーバー（ML 依存は gpu グループに分離）
├── robot/                    # 既存の診断スクリプト（段階的に robot_app へ移す）
├── pc_server/                # 既存サーバー（段階的に server へ移し、最後に削除する）
├── deploy/                   # robot / server の systemd ユニット・デプロイスクリプト
└── docs/
```
- ロボットの実行環境は既存の `/venvs/apps_venv`（reachy_mini SDK 入り）。`robot_app` と `protocol` をそこへインストールする。
- 既存コードは一括で書き換えず、Issue ごとに「動く状態を保ったまま 1 歩ずつ」移す。移行中は今の `pc_server` と `robot/conversation_ws.py` が動き続ける。

---

## 7. CI と開発フロー

- **GitHub Actions**（PR と main への push で実行）
  - `lock`: `uv lock --check`
  - `lint`: ruff check / ruff format --check
  - `typecheck`: pyright（`protocol` と判断系は strict から始める）
  - `test-protocol` / `test-robot` / `test-server`: パッケージごとの pytest。サーバーの ML 依存（torch・faster-whisper・SBV2）は入れない
  - `contract`: Fake プロバイダで動かした本物のサーバーにロボット側クライアントをつなぎ、シナリオを流す
  - （将来）GPU 結合テスト: 研究室 GPU マシンをセルフホストランナーにし、手動または夜間に実行する
- **ブランチ運用**: `main` は保護（PR 必須・CI 必須）。ブランチ名は `feat/<issue#>-<slug>`、PR 本文に `Closes #<issue>`。
- **開発サイクル**: Claude が Issue を取る → ブランチ → 実装とテスト → PR → オーナーがレビューしてマージ → 次の Issue。
- **実機検証**: CI ではできないので、PR テンプレートに「実機確認」欄を設け、必要な PR はオーナーが実機で確認する。

---

## 8. マイルストーン

| マイルストーン | 内容 | 完了条件 |
|---|---|---|
| **M0 基盤整備** | 現行コードの main 取り込み、CI、テスト基盤、テンプレート | main が CI グリーン、保護ルールが有効 |
| **M1 プロトコルとリポジトリ再編** | uv workspace 化、`protocol` v1、フレーミング、契約テストの枠組み、パッケージ別 CI | protocol のテストと契約テストの雛形が CI で通る |
| **M2 サーバー** | サーバーの骨格、セッションの状態機械、STT/LLM/TTS プロバイダ、会話履歴、memory.op、デプロイ | Fake 構成と GPU 実機の両方で 1 ターンが通る |
| **M3 ロボットのフロントエンド基盤** | 骨格、イベントバス、RobotIO、World、サーバークライアント、フォールバック、記録と再生、systemd | 新しいサーバーと実機で会話できる（旧構成と同等） |
| **M4 知覚** | 多人数トラッカー、ロボット上の顔同定、視線、DoA と顔の対応、入室検知 | 入室・通過・退出が実機で妥当に判定される |
| **M5 記憶** | SQLite、移行、来訪ログ、親密度、プロフィール、memory.op の適用、要約、忘却 | 親密度によって挨拶が変わる |
| **M6 社会的判断** | シナリオ評価、FloorManager、分類 API、InitiativePolicy、BargeIn、相槌 | シナリオ評価が CI で全件通る |
| **M7 エージェント刷新** | StateGraph、構造化コンテキスト、ペルソナ、ガード、感情タグ、ツール RPC | 評価シナリオで品質を確認 |
| **M8 表現** | 感情→モーション、モーション合成、所作の追加、フィラー | 実機で発話と動作が同期する |
| **M9 運用と評価** | LLM 評価ワークフロー、メトリクス、GPU セルフホストランナー、1 週間の連続運用 | 1 週間止まらずに動く |

Issue はマイルストーンごとに Epic（親 Issue）を立て、作業単位をサブイシューとしてぶら下げる。

---

## 9. 未決事項（オーナーの判断が必要）

1. **LLM**: 当面はサーバーからクラウド API（OpenAI）を呼ぶ前提。GPU マシンでローカル LLM に置き換えるかは M7 以降に評価する。
2. **GPU マシンの仕様**: GPU の型と VRAM（STT・SBV2・将来のローカル LLM を同時に載せられるか）。M2 のベンチマークで確認する。
3. **研究室メンバーの情報**: 名前を事前に登録するか、会話で覚えさせるか（初期は会話で覚える）。
4. **掲示と同意**: 顔を覚えることの掲示と、オプトアウトの方法。
5. **場の文脈バッファ**: 自分宛てでない会話を短期だけ保持する機能を有効にするか（既定は 10 分・非永続）。
