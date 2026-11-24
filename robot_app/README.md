# robot_app（reachy_front）

Reachy Mini 上で動くフロントエンド。知覚・判断・記憶・サーバー呼び出し・表現を担う（[設計書 §3](../docs/DESIGN.md)）。

## モジュール
`core/`（設定・イベントバス・時計・ログ）、`io/`、`perception/`、`world/`、`social/`、`memory/`、`client/`、`expression/`、`harness/`。中身は Issue ごとに足していく（骨格だけが先にある）。

## 設定
環境変数で読む（`reachy_front.core.FrontSettings`）。名前は移行前の `robot/` と同じなので、今の起動スクリプトをそのまま使える。

| 区分 | 環境変数 |
|---|---|
| アイコンタクト | `EC_FRONTAL_MAX` `EC_LEVEL_MAX` `EC_FACE_MIN_FRAC` `EC_CENTER_MAX` `EC_VERT_MIN` `EC_VERT_MAX` `EC_GRACE` |
| 自発発話 | `PROACTIVE_ENABLED` `PROACTIVE_ENGAGE_SEC` `PROACTIVE_MIN_GAP` `PROACTIVE_POLL_SEC` |
| 宛先の条件 | `GATE_EYE_SEC` |
| 視線 | `GAZE_ENABLED` `GAZE_KP_YAW` `GAZE_KP_PITCH` `GAZE_YAW_SIGN` `GAZE_PITCH_SIGN` `GAZE_DEBUG` |
| サーバー | `PC_HOST` `PC_PORT` `REACHY_ROBOT_ID`（既定 reachy-mini）`REACHY_SERVER_TOKEN`（16 文字以上。サーバーと同じ値） |

起動すると、設定をログに 1 行で出す（トークンは値を出さない）: `python -m reachy_front`

## 実機（apps_venv）への入れ方
実機の Python は `/venvs/apps_venv`（reachy_mini SDK 入り）。`protocol/` と `robot_app/` を rsync して、その環境へ入れる。
```bash
rsync -avz --exclude __pycache__ protocol robot_app pollen@reachy-mini.local:/home/pollen/reachy-mini-chat/
ssh pollen@reachy-mini.local '/venvs/apps_venv/bin/pip install /home/pollen/reachy-mini-chat/protocol /home/pollen/reachy-mini-chat/robot_app'
ssh pollen@reachy-mini.local 'PC_HOST=<PC のホスト名>.local /venvs/apps_venv/bin/python -m reachy_front'
```
`pydantic` と `pydantic-settings` が apps_venv に無ければ、上の pip が一緒に入れる（ロボットがネットに出られること）。**この手順は実機でまだ試していない**（実機確認はロボットが使えるときに行う）。
