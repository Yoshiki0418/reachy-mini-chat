"""バイナリフレーム: 往復・kind と format の組み合わせ・壊れたデータの拒否。"""

from __future__ import annotations

import json
import struct

import pytest

from reachy_protocol.framing import (
    MAX_FRAME_BYTES,
    MAX_HEADER_BYTES,
    BinaryHeader,
    FramingError,
    decode_frame,
    encode_frame,
)

PCM = struct.pack("<4f", 0.0, 0.5, -0.5, 1.0)  # float32 little-endian 4 サンプル
WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "  # 中身は検査しない（形式名で区別する）
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16

AUDIO_IN = BinaryHeader(kind="audio_in", ref="u1", seq=3, format="pcm_f32le", sr=16000)
AUDIO_OUT = BinaryHeader(kind="audio_out", ref="t1", seq=0, format="wav_pcm16")
FRAME = BinaryHeader(kind="frame", ref="u1", seq=0, format="jpeg")


def _raw(header: dict[str, object], payload: bytes = b"") -> bytes:
    head = json.dumps(header).encode()
    return struct.pack(">I", len(head)) + head + payload


@pytest.mark.parametrize(
    ("header", "payload"),
    [(AUDIO_IN, PCM), (AUDIO_OUT, WAV), (FRAME, JPEG), (AUDIO_OUT, b"")],
    ids=["audio_in", "audio_out", "frame", "empty-payload"],
)
def test_round_trip(header: BinaryHeader, payload: bytes) -> None:
    got_header, got_payload = decode_frame(encode_frame(header, payload))
    assert got_header == header
    assert got_payload == payload


def test_layout_is_length_prefixed_json() -> None:
    data = encode_frame(AUDIO_OUT, WAV)
    (head_len,) = struct.unpack(">I", data[:4])
    assert json.loads(data[4 : 4 + head_len]) == {
        "kind": "audio_out",
        "ref": "t1",
        "seq": 0,
        "format": "wav_pcm16",
    }
    assert data[4 + head_len :] == WAV


def test_payload_may_contain_anything() -> None:
    """ペイロードに JSON 風のバイト列や区切りが含まれても影響しない。"""
    payload = b'{"kind":"x"}' + b"\x00\x00\x00\x10" + bytes(range(256))
    assert decode_frame(encode_frame(FRAME, payload))[1] == payload


# ---------------------------------------------------------------- ヘッダの規約


@pytest.mark.parametrize(
    ("kind", "fmt"),
    [
        ("audio_in", "wav_pcm16"),
        ("audio_out", "pcm_f32le"),
        ("frame", "wav_pcm16"),
        ("audio_out", "jpeg"),
    ],
)
def test_kind_and_format_must_match(kind: str, fmt: str) -> None:
    with pytest.raises(ValueError, match="format"):
        BinaryHeader.model_validate(
            {"kind": kind, "ref": "r", "seq": 0, "format": fmt, "sr": 16000}
        )


def test_pcm_needs_sample_rate() -> None:
    with pytest.raises(ValueError, match="sr"):
        BinaryHeader(kind="audio_in", ref="u1", seq=0, format="pcm_f32le")


@pytest.mark.parametrize("bad", [{"seq": -1}, {"ref": ""}, {"sr": 0}])
def test_header_field_bounds(bad: dict[str, object]) -> None:
    base: dict[str, object] = {
        "kind": "audio_in",
        "ref": "u1",
        "seq": 0,
        "format": "pcm_f32le",
        "sr": 16000,
    }
    with pytest.raises(ValueError):
        BinaryHeader.model_validate(base | bad)


def test_unknown_header_fields_are_ignored() -> None:
    data = _raw({"kind": "frame", "ref": "u1", "seq": 0, "format": "jpeg", "w": 512})
    assert decode_frame(data)[0] == FRAME


# ---------------------------------------------------------------- 壊れたデータ


def test_pcm_length_must_be_multiple_of_four() -> None:
    with pytest.raises(FramingError, match="4 の倍数"):
        encode_frame(AUDIO_IN, PCM[:-1])
    with pytest.raises(FramingError, match="4 の倍数"):
        decode_frame(_raw(AUDIO_IN.model_dump(exclude_none=True), PCM[:-1]))


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"", "短すぎる"),
        (b"\x00\x00", "短すぎる"),
        (struct.pack(">I", 0), "ヘッダ長"),
        (struct.pack(">I", MAX_HEADER_BYTES + 1) + b"x" * 10, "ヘッダ長"),
        (struct.pack(">I", 100) + b'{"kind":', "途中で切れて"),
        (struct.pack(">I", 3) + b"\xff\xfe\xfd", "ヘッダが不正"),  # UTF-8 でない
        (struct.pack(">I", 5) + b"{oops", "ヘッダが不正"),  # JSON でない
        (_raw({"kind": "nope", "ref": "r", "seq": 0, "format": "jpeg"}), "ヘッダが不正"),
        (_raw({"kind": "frame", "ref": "r", "seq": 0, "format": "wav_pcm16"}), "ヘッダが不正"),
    ],
    ids=[
        "empty",
        "short",
        "zero-header",
        "huge-header",
        "truncated",
        "not-utf8",
        "not-json",
        "bad-kind",
        "bad-combo",
    ],
)
def test_corrupt_frames_raise_framing_error(data: bytes, match: str) -> None:
    with pytest.raises(FramingError, match=match):
        decode_frame(data)


def test_frame_size_limit() -> None:
    too_big = b"\x00" * MAX_FRAME_BYTES
    with pytest.raises(FramingError, match="大きすぎる"):
        encode_frame(FRAME, too_big)
    with pytest.raises(FramingError, match="大きすぎる"):
        decode_frame(too_big + b"\x00")


def test_framing_error_is_value_error() -> None:
    """受信ループで ValueError をまとめて扱えるように。"""
    assert issubclass(FramingError, ValueError)


def test_header_size_limit_on_encode(monkeypatch: pytest.MonkeyPatch) -> None:
    """今のフィールドでは超えないが、将来ヘッダに項目を足したときのための上限。"""
    import reachy_protocol.framing as framing

    monkeypatch.setattr(framing, "MAX_HEADER_BYTES", 10)
    with pytest.raises(FramingError, match="ヘッダが大きすぎる"):
        framing.encode_frame(FRAME, JPEG)
