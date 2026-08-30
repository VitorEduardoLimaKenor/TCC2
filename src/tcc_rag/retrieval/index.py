"""Índice vetorial no Qdrant (Seção 4.9).

Dois modos, mesma interface:

* ``local`` — armazenamento em arquivo, sem servidor. É o modo a usar no Colab
  e no Kaggle; basta guardar o diretório no Drive para o índice sobreviver
  entre sessões.
* ``server`` — contêiner Docker, útil no desenvolvimento local.

A filtragem por metadados é o recurso que motiva a escolha do Qdrant: a busca é
restrita ao banco e ao idioma da pergunta de teste, e à partição do pool.
"""

from __future__ import annotations

import uuid
from typing import Any, Iterable

import numpy as np

from ..config import QdrantConfig
from ..schemas import Demo


def _point_id(uid: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, uid))


class DemoIndex:
    def __init__(self, cfg: QdrantConfig, dimension: int):
        from qdrant_client import QdrantClient

        self.cfg = cfg
        self.dimension = dimension
        if cfg.mode == "local":
            self.client = QdrantClient(path=cfg.path)
        else:
            self.client = QdrantClient(url=cfg.url)

    # ------------------------------------------------------------ criação

    def recreate(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        distance = getattr(Distance, self.cfg.distance.upper(), Distance.COSINE)
        if self.client.collection_exists(self.cfg.collection):
            self.client.delete_collection(self.cfg.collection)
        self.client.create_collection(
            collection_name=self.cfg.collection,
            vectors_config=VectorParams(size=self.dimension, distance=distance),
        )
        for field in ("db_id", "lang", "partition", "family_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.cfg.collection,
                    field_name=field,
                    field_schema="keyword",
                )
            except Exception:
                # O modo local aceita filtros sem índice explícito.
                pass

    def upsert(self, demos: Iterable[Demo], vectors: np.ndarray, batch: int = 256) -> int:
        from qdrant_client.models import PointStruct

        demos = list(demos)
        assert len(demos) == len(vectors), "demos e vetores com tamanhos diferentes"

        total = 0
        for start in range(0, len(demos), batch):
            chunk = demos[start : start + batch]
            points = [
                PointStruct(
                    id=_point_id(demo.uid),
                    vector=vectors[start + offset].tolist(),
                    payload=demo.payload(),
                )
                for offset, demo in enumerate(chunk)
            ]
            self.client.upsert(collection_name=self.cfg.collection, points=points)
            total += len(points)
        return total

    # -------------------------------------------------------------- busca

    def search(
        self,
        vector: np.ndarray,
        db_id: str,
        lang: str,
        partition: str = "pool",
        limit: int = 10,
        exclude_family: str | None = None,
    ) -> list[tuple[Demo, float]]:
        """Vizinhos mais próximos, restritos ao banco, ao idioma e ao pool.

        `exclude_family` existe para diagnóstico: permite medir quanto do
        desempenho vem de famílias idênticas caso o particionamento falhe.
        Na execução normal ele é None, porque a partição já garante isso.
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        must = [
            FieldCondition(key="db_id", match=MatchValue(value=db_id)),
            FieldCondition(key="lang", match=MatchValue(value=lang)),
            FieldCondition(key="partition", match=MatchValue(value=partition)),
        ]
        must_not = []
        if exclude_family:
            must_not.append(
                FieldCondition(key="family_id", match=MatchValue(value=exclude_family))
            )

        hits = self.client.query_points(
            collection_name=self.cfg.collection,
            query=vector.tolist(),
            query_filter=Filter(must=must, must_not=must_not or None),
            limit=limit,
            with_payload=True,
        ).points

        return [(Demo.from_payload(hit.payload), float(hit.score)) for hit in hits]

    def count(self, **filters: Any) -> int:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in filters.items()
        ]
        result = self.client.count(
            collection_name=self.cfg.collection,
            count_filter=Filter(must=conditions) if conditions else None,
            exact=True,
        )
        return int(result.count)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass
