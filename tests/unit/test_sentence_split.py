from pc_server.pipeline import _is_name_call, _split_completed_sentences


class TestSplitCompletedSentences:
    def test_empty(self) -> None:
        assert _split_completed_sentences("") == ([], "")

    def test_no_terminator_keeps_remainder(self) -> None:
        assert _split_completed_sentences("こんにちは") == ([], "こんにちは")

    def test_multiple_sentences_and_tail(self) -> None:
        sentences, rest = _split_completed_sentences("おはよう。元気？今日は")
        assert sentences == ["おはよう。", "元気？"]
        assert rest == "今日は"

    def test_ascii_and_newline_terminators(self) -> None:
        sentences, rest = _split_completed_sentences("Hi! OK.\nnext")
        assert sentences == ["Hi!", "OK."]
        assert rest == "next"

    def test_skips_whitespace_only_pieces(self) -> None:
        sentences, rest = _split_completed_sentences("。 。")
        assert sentences == ["。", "。"]
        assert rest == ""


class TestIsNameCall:
    def test_katakana(self) -> None:
        assert _is_name_call("リーチ、おはよう")

    def test_hiragana_and_stt_variants(self) -> None:
        assert _is_name_call("ねえりーち")
        assert _is_name_call("レイチ聞いて")

    def test_english_case_insensitive(self) -> None:
        assert _is_name_call("Hey Reachy")

    def test_bare_reach_is_not_name(self) -> None:
        # 英語の幻聴 "reach" には反応しない
        assert not _is_name_call("I can reach it")

    def test_unrelated(self) -> None:
        assert not _is_name_call("今日のゼミ何時から？")
