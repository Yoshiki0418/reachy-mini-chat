"""Style-Bert-VITS2 TTSラッパー (Mac PC用)。

学長モデル等を読み込み、テキスト→WAV(numpy)を返す。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Monkeypatch: 一部環境でBERT出力がFP16になりLinear層(FP32)とtype mismatchになる対策。
# import 順序が重要 (TTSModel より先に当てる)
import style_bert_vits2.models.infer as infer_module

_original_get_text = infer_module.get_text


def _get_text_patched(*args, **kwargs):
    bert, ja_bert, en_bert, phone, tone, language = _original_get_text(*args, **kwargs)
    return (
        bert.float(),
        ja_bert.float(),
        en_bert.float(),
        phone,
        tone,
        language,
    )


infer_module.get_text = _get_text_patched

from style_bert_vits2.constants import (  # noqa: E402
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from style_bert_vits2.nlp import bert_models  # noqa: E402
from style_bert_vits2.tts_model import TTSModel  # noqa: E402

# 日本語用 DeBERTa BERT モデル/トークナイザ。未配置なら HuggingFace Hub から取得する。
# gakucho は Linux の絶対パス直書きだが、macOS/クロスプラットフォームのため repo id で解決する。
JP_BERT_REPO = "ku-nlp/deberta-v2-large-japanese-char-wwm"
_bert_ready = False


def _ensure_jp_bert() -> None:
    """日本語 BERT モデル/トークナイザをロード (初回は HF からダウンロード)。"""
    global _bert_ready
    if _bert_ready:
        return
    bert_models.load_model(Languages.JP, JP_BERT_REPO)
    bert_models.load_tokenizer(Languages.JP, JP_BERT_REPO)
    _bert_ready = True


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon の MPS は一部演算で落ちることがあるので CPU を既定にする
    return "cpu"


class StyleBertVITS2TTS:
    def __init__(
        self,
        model_dir: str | Path,
        device: str | None = None,
        length: float = 1.1,
    ) -> None:
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        style_vec_path = model_dir / "style_vectors.npy"

        ckpts = sorted(
            model_dir.glob("G_*.pth"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not ckpts:
            raise FileNotFoundError(f"{model_dir} に G_*.pth が見つかりません。")
        model_path = ckpts[0]

        for p in (model_path, config_path, style_vec_path):
            if not p.exists():
                raise FileNotFoundError(f"必須ファイルがありません: {p}")

        if device is None:
            device = _pick_device()

        _ensure_jp_bert()

        self.length = length
        self.tts_model = TTSModel(
            model_path=model_path,
            config_path=config_path,
            style_vec_path=style_vec_path,
            device=device,
        )
        print(
            f"[SBV2] loading {model_path.name} on {device}…",
            flush=True,
            file=sys.stderr,
        )
        self.tts_model.load()
        print("[SBV2] loaded.", flush=True, file=sys.stderr)

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """テキストを合成し (sample_rate, float32 [-1, 1] mono) を返す。

        SBV2 の infer は int16 を返すため、ここで float32 [-1, 1] に正規化する。
        """
        if not text.strip():
            raise ValueError("テキストが空です")
        sr, audio = self.tts_model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=0,
            sdp_ratio=DEFAULT_SDP_RATIO,
            noise=DEFAULT_NOISE,
            noise_w=DEFAULT_NOISEW,
            length=self.length,
        )
        audio = np.asarray(audio)
        if audio.dtype.kind == "i":
            audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
        else:
            audio = audio.astype(np.float32)
        return sr, audio
