"""応答テキストの文分割。LLM のストリームを文単位で TTS に流すために使う。"""

from __future__ import annotations

SENTENCE_END = "。.!?！？\n"


def split_completed(buffer: str) -> tuple[list[str], str]:
    """buffer から区切り文字で終わる完結文を切り出す。戻り値は (完結文のリスト, 未完の残り)。

    空白だけの断片は捨てる。区切り文字が連続する場合（「！？」など）は 1 文にまとめる。
    """
    sentences: list[str] = []
    start = 0
    i = 0
    n = len(buffer)
    while i < n:
        if buffer[i] in SENTENCE_END:
            j = i + 1
            while j < n and buffer[j] in SENTENCE_END and buffer[j] != "\n":
                j += 1
            piece = buffer[start:j].strip()
            if piece:
                sentences.append(piece)
            start = i = j
        else:
            i += 1
    return sentences, buffer[start:]
