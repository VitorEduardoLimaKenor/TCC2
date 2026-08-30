"""Vetorização com cache em disco.

O modelo é multilíngue por exigência do desenho experimental (Seção 4.3.1): um
codificador predominantemente inglês introduziria, no recuperador, um viés que
se confundiria com o efeito linguístico que o trabalho quer medir.

O cache é indispensável na fase de validação, que varre a grade de k, N e λ
sobre o mesmo conjunto de perguntas dezenas de vezes.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from ..config import EmbeddingConfig


class Embedder:
    def __init__(self, cfg: EmbeddingConfig, cache_dir: str | Path):
        self.cfg = cfg
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._mem: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ interno

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": self.cfg.device}
            if self.cfg.revision:
                kwargs["revision"] = self.cfg.revision
            self._model = SentenceTransformer(self.cfg.model, **kwargs)
        return self._model

    @property
    def _namespace(self) -> str:
        raw = f"{self.cfg.model}@{self.cfg.revision or 'main'}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    def _cache_path(self, kind: str) -> Path:
        return self.cache_dir / f"emb_{self._namespace}_{kind}.pkl"

    def _key(self, text: str, kind: str) -> str:
        return hashlib.sha1(f"{kind}||{text}".encode("utf-8")).hexdigest()

    def _load(self, kind: str) -> dict[str, np.ndarray]:
        path = self._cache_path(kind)
        if path.exists():
            with path.open("rb") as fh:
                return pickle.load(fh)
        return {}

    def _save(self, kind: str, data: dict[str, np.ndarray]) -> None:
        with self._cache_path(kind).open("wb") as fh:
            pickle.dump(data, fh, protocol=pickle.HIGHEST_PROTOCOL)

    # -------------------------------------------------------------- API

    def encode(self, texts: list[str], kind: str = "query") -> np.ndarray:
        """Vetoriza uma lista de textos.

        `kind` distingue consulta de documento porque a família E5 exige
        prefixos diferentes para cada papel; com BGE-m3 basta deixar os
        prefixos vazios na configuração.
        """
        prefix = self.cfg.query_prefix if kind == "query" else self.cfg.passage_prefix
        cache = self._load(kind)

        keys = [self._key(t, kind) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in cache]

        if missing_idx:
            batch = [prefix + texts[i] for i in missing_idx]
            vectors = self.model.encode(
                batch,
                batch_size=self.cfg.batch_size,
                normalize_embeddings=self.cfg.normalize,
                show_progress_bar=len(batch) > 256,
                convert_to_numpy=True,
            )
            for pos, i in enumerate(missing_idx):
                cache[keys[i]] = vectors[pos].astype(np.float32)
            self._save(kind, cache)

        return np.stack([cache[k] for k in keys])

    def encode_one(self, text: str, kind: str = "query") -> np.ndarray:
        return self.encode([text], kind=kind)[0]

    @property
    def dimension(self) -> int:
        # O nome mudou nas versões recentes do sentence-transformers.
        for nome in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
            metodo = getattr(self.model, nome, None)
            if callable(metodo):
                return int(metodo())
        raise AttributeError("modelo sem método de dimensão de embedding")
