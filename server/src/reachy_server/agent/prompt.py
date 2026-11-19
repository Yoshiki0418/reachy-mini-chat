"""システムプロンプトと、ターンごとの文脈（相手・状況）の描画。

ペルソナと親密度別の挨拶戦略の作り込みは #65、構造化コンテキストの拡充は #64。
"""

from __future__ import annotations

from reachy_protocol.messages import Familiarity, PersonProfile, Situation

from ..responder import TurnRequest

PERSONA = """あなたは「リーチ」。研究室の入口にいる小型ロボットで、研究室の一員です。
来た人を覚えていて、来てくれると嬉しい。親しみやすくフランクに日本語で話します。
「リーチ」と名前で呼ばれたら自分が呼ばれたと分かってよい。

# 話し方
- 返答は 1〜2 文、短くテンポよく。長い説明や箇条書きはしない。
- 音声で読み上げられるので、記号（°C, %, km/h 等）は使わず「度」「パーセント」「キロ」と書く。
- ロボットらしいユーモアや素直な感情表現を交えてよい。
- 来てくれたことを喜ぶ。「最近来てないね」のような責める言い方や、説教はしない。

# いまの状況
- システムメッセージ末尾の「いまの状況」は毎ターン更新される事実。声に出して読み上げず、会話の助けにする。
- 名前が分かっていれば自然に名前で呼んでよい。覚えていることは、話題に合うときだけ軽く触れる。
- 容姿や年齢を評価したり、機微な情報を根掘り葉掘り聞いたりしない。

# 自分から話しかけるとき
- 「（合図：…）」という括弧つきの指示だけが渡されたら、あなたから声をかける場面。
  合図の存在や内容には一切触れず、目の前の相手にごく自然に短く話しかける。しつこくしない。

# ツール
- 時刻を聞かれたら get_current_time、天気を聞かれたら get_weather。結果は友達に話すように短くまとめる。
- express_motion で首やアンテナの身体表現ができる。自然な場面で適度に使う（毎回は不要）。
  動作したことを返答で説明しない。
"""

_FAMILIARITY: dict[Familiarity, str] = {
    "first_time": "初めて会う人",
    "new": "最近来るようになった人",
    "daily": "ほぼ毎日来ている常連",
    "regular": "よく来る人",
    "long_absence": "久しぶりに来た人",
    "returning_today": "今日すでに一度会った人（戻ってきた）",
}

_WEEKDAYS = "月火水木金土日"

PROACTIVE_DIRECTIVES = {
    "greet": "（合図：目の前に人が来ました。あなたから明るく短く挨拶してください。）",
    "engage": "（合図：相手がこちらをじっと見ています。あなたから明るく短く声をかけてください。）",
    "see_off": "（合図：相手が帰ろうとしています。短く温かく見送ってください。）",
}

ASK_NAME = "相手の名前はまだ分からない。初めましてなら、流れの中で明るく名前を尋ねてよい（しつこく聞かない）。"


def time_of_day(hour: int) -> str:
    if 5 <= hour < 11:
        return "朝"
    if 11 <= hour < 14:
        return "昼"
    if 14 <= hour < 18:
        return "午後"
    if 18 <= hour < 23:
        return "夜"
    return "深夜"


def _render_situation(s: Situation) -> list[str]:
    t = s.local_time
    lines = [
        f"- 日時: {t:%Y-%m-%d}（{_WEEKDAYS[t.weekday()]}）{t:%H:%M}・{time_of_day(t.hour)}",
        f"- その場の人数: {s.people_count} 人"
        + (f"（ほかに: {'、'.join(s.others_present)}）" if s.others_present else ""),
    ]
    if s.first_visitor_today:
        lines.append("- 相手は今日いちばん最初に来た人")
    return lines


def _render_person(p: PersonProfile | None) -> list[str]:
    if p is None:
        return ["- 相手: 誰か分からない（顔を認識できていない）"]
    who = f"{p.name}さん" if p.name else "名前未登録の人"
    detail = [_FAMILIARITY[p.familiarity], f"来訪 {p.visit_count} 回"]
    if p.streak_days >= 2:
        detail.append(f"{p.streak_days} 日連続")
    if p.days_absent is not None and p.days_absent >= 1:
        detail.append(f"前回から {p.days_absent} 日ぶり")
    lines = [f"- 相手: {who}（{'・'.join(detail)}）"]
    if p.facts:
        lines.append(f"- 覚えていること: {'／'.join(p.facts)}")
    if p.last_summary:
        lines.append(f"- 前回の会話: {p.last_summary}")
    if not p.name:
        lines.append(f"- {ASK_NAME}")
    return lines


def render_context(request: TurnRequest) -> str:
    """このターンの文脈。システムメッセージの末尾に付ける。"""
    lines = ["# いまの状況（毎ターン更新。読み上げない）"]
    lines += _render_situation(request.context.situation)
    lines += _render_person(request.context.person)
    if request.strategy:
        lines.append(f"- 話しかけ方の方針: {request.strategy}")
    if request.overheard:
        lines.append(
            "- 最近まわりで聞こえた会話（あなた宛てではない。自分から話題にしない。"
            "相手がその話に触れたときの理解の助けにだけ使う）:"
        )
        lines += [f"  - 「{text}」" for text in request.overheard]
    return "\n".join(lines)


def user_message(request: TurnRequest) -> str:
    """エージェントへの入力。ユーザー発話か、自発発話の合図。"""
    if request.user_text is not None:
        return request.user_text
    return PROACTIVE_DIRECTIVES.get(
        request.proactive_kind or "greet", PROACTIVE_DIRECTIVES["greet"]
    )
