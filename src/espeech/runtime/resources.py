from __future__ import annotations

import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from f5_tts.infer.utils_infer import load_model, load_vocoder
from f5_tts.model import DiT
from huggingface_hub import hf_hub_download, snapshot_download
from ruaccent import RUAccent

from espeech.config import MODEL_CFG, MODEL_FILE, MODEL_REPO, VOCAB_FILE
from espeech.runtime.compatibility import (
    patch_ruaccent_token_type_ids,
    patch_torchaudio_load_if_needed,
)


@dataclass(slots=True)
class ResourceManager:
    log: Callable[[str], None] | None = print
    hf_token: str | None = field(default_factory=lambda: os.getenv("HF_TOKEN"))
    _model: Any = field(default=None, init=False, repr=False)
    _accentizer: RUAccent | None = field(default=None, init=False, repr=False)
    _vocoder: Any = field(default=None, init=False, repr=False)
    inference_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        patch_torchaudio_load_if_needed(logger=self.log)

    def _log(self, message: str) -> None:
        if self.log is not None:
            self.log(message)

    def _download_model_artifacts(self) -> tuple[str, str]:
        model_path = None
        vocab_path = None

        self._log(
            f"Trying to download model file '{MODEL_FILE}' and '{VOCAB_FILE}' from hub '{MODEL_REPO}'"
        )
        try:
            model_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                token=self.hf_token,
            )
            vocab_path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=VOCAB_FILE,
                token=self.hf_token,
            )
            self._log(f"Downloaded model to {model_path}")
            self._log(f"Downloaded vocab to {vocab_path}")
        except Exception as exc:
            self._log(f"hf_hub_download failed: {exc}")

        if model_path is None or vocab_path is None:
            try:
                local_dir = Path(f"cache_{MODEL_REPO.replace('/', '_')}")
                self._log(f"Attempting snapshot_download into {local_dir}...")
                snapshot_dir = snapshot_download(
                    repo_id=MODEL_REPO,
                    cache_dir=None,
                    local_dir=str(local_dir),
                    token=self.hf_token,
                )
                possible_model = Path(snapshot_dir) / MODEL_FILE
                possible_vocab = Path(snapshot_dir) / VOCAB_FILE
                if possible_model.exists():
                    model_path = str(possible_model)
                if possible_vocab.exists():
                    vocab_path = str(possible_vocab)
                self._log(f"Snapshot downloaded to {snapshot_dir}")
            except Exception as exc:
                self._log(f"snapshot_download failed: {exc}")

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model file not found after download attempts: {model_path}"
            )
        if not vocab_path or not Path(vocab_path).exists():
            raise FileNotFoundError(
                f"Vocab file not found after download attempts: {vocab_path}"
            )

        return model_path, vocab_path

    def ensure_model(self):
        if self._model is not None:
            return self._model

        model_path, vocab_path = self._download_model_artifacts()
        self._log(f"Loading model from: {model_path}")
        self._model = load_model(DiT, MODEL_CFG, model_path, vocab_file=vocab_path)
        return self._model

    def ensure_accentizer(self) -> RUAccent:
        if self._accentizer is not None:
            return self._accentizer

        self._log("Loading RUAccent...")
        accentizer = RUAccent()
        accentizer.load(
            omograph_model_size="turbo3.1",
            use_dictionary=True,
            tiny_mode=False,
        )
        patch_ruaccent_token_type_ids(accentizer)
        self._accentizer = accentizer
        self._log("RUAccent loaded.")
        return self._accentizer

    def ensure_vocoder(self):
        if self._vocoder is not None:
            return self._vocoder

        self._log("Loading vocoder...")
        self._vocoder = load_vocoder()
        self._log("Vocoder loaded.")
        return self._vocoder
