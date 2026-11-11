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

import json
import uuid
from pathlib import Path
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
        self._abrir()

    def _abrir(self) -> None:
        from qdrant_client import QdrantClient

        if self.cfg.mode == "local":
            self.client = QdrantClient(path=self.cfg.path)
        else:
            self.client = QdrantClient(url=self.cfg.url)

    def reabrir(self) -> None:
        """Fecha e reabre o cliente: no modo local, relê a coleção do disco."""
        self.close()
        self._abrir()

    # ---------------------------------------------------------- identidade

    @property
    def identidade_path(self) -> Path:
        """Arquivo ao lado do diretório do Qdrant, fora do que o cliente gerencia."""
        base = Path(self.cfg.path)
        return base.with_name(f"{base.name}.{self.cfg.collection}.identidade.json")

    def gravar_identidade(self, identidade: dict) -> None:
        """Registra com que pool e vetorizador o índice foi construído (`pipeline.identidade_indice`)."""
        self.identidade_path.parent.mkdir(parents=True, exist_ok=True)
        self.identidade_path.write_text(
            json.dumps(identidade, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8"
        )

    @property
    def referencia_path(self) -> Path:
        base = Path(self.cfg.path)
        return base.with_name(f"{base.name}.{self.cfg.collection}.vetores.npz")

    TOLERANCIA_VETORES = 1e-5
    """Diferença máxima, por componente, entre o índice e a referência gravada.

    O Qdrant local normaliza os vetores ao carregar a coleção, e o resultado
    difere do estado em memória logo após o `upsert` (e pode diferir entre
    processadores) na ordem de 1e-8. Um hash dos vetores lidos do índice, mesmo
    arredondado, muda com esse ruído; por isso a conferência é por tolerância
    contra uma referência gravada por `index`, e o hash é o da referência.
    """

    def _ler_vetores(self) -> tuple[list[str], np.ndarray]:
        """Vetores do próprio índice, em float32, ordenados por uid."""
        vetores = {}
        offset = None
        while True:
            pontos, offset = self.client.scroll(
                collection_name=self.cfg.collection, limit=512, offset=offset,
                with_payload=["uid"], with_vectors=True,
            )
            for ponto in pontos:
                vetores[ponto.payload["uid"]] = np.asarray(ponto.vector, dtype=np.float32)
            if offset is None:
                break
        uids = sorted(vetores)
        matriz = np.stack([vetores[u] for u in uids]) if uids else np.zeros((0, self.dimension), np.float32)
        return uids, matriz

    @staticmethod
    def _sha_vetores(uids: list[str], matriz: np.ndarray) -> str:
        import hashlib

        sha = hashlib.sha256()
        for uid, vetor in zip(uids, matriz):
            sha.update(uid.encode("utf-8") + b"\x1f" + np.ascontiguousarray(vetor, dtype=np.float32).tobytes())
        return sha.hexdigest()

    def gravar_referencia(self) -> str:
        """Grava os vetores do índice como referência e devolve o SHA-256 dela.

        Chamado por `index` logo depois do `upsert`. O cliente é reaberto
        antes da leitura: no modo local, os vetores em memória logo após o
        `upsert` não passaram pela normalização feita na carga, e a referência
        precisa descrever o índice como as execuções vão lê-lo. A referência
        fica ao lado do diretório do Qdrant e viaja com ele.
        """
        self.reabrir()
        uids, matriz = self._ler_vetores()
        self.referencia_path.parent.mkdir(parents=True, exist_ok=True)
        with self.referencia_path.open("wb") as fh:
            np.savez(fh, uids=np.array(uids), vetores=matriz)
        return self._sha_vetores(uids, matriz)

    def vetores_sha256(self) -> str:
        """Identidade dos vetores do índice.

        Lê o próprio índice: o digest descreve o que a busca vai usar, não o
        que se esperava gravar. Havendo referência gravada por `index`, o
        índice é comparado a ela por tolerância e, coincidindo, o SHA devolvido
        é o da referência, estável entre sessões e máquinas. Sem referência,
        ou se o índice divergir dela, o SHA é o dos vetores lidos.
        """
        uids, matriz = self._ler_vetores()
        if self.referencia_path.exists():
            with np.load(self.referencia_path, allow_pickle=False) as ref:
                ref_uids, ref_matriz = [str(u) for u in ref["uids"]], ref["vetores"]
            if (ref_uids == uids and ref_matriz.shape == matriz.shape
                    and np.allclose(ref_matriz, matriz, rtol=0.0, atol=self.TOLERANCIA_VETORES)):
                return self._sha_vetores(ref_uids, ref_matriz)
        return self._sha_vetores(uids, matriz)

    def ler_identidade(self) -> dict | None:
        if not self.identidade_path.exists():
            return None
        return json.loads(self.identidade_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------ criação

    def recreate(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        # A identidade antiga deixa de valer assim que a coleção é apagada; se a
        # construção cair no meio, o índice fica sem identidade e é recusado.
        self.identidade_path.unlink(missing_ok=True)
        self.referencia_path.unlink(missing_ok=True)
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
