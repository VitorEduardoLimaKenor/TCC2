"""Correspondência exata e dificuldade oficiais do Spider (E03).

A referência de cada caso é o próprio `evaluation.evaluate` oficial, com
`etype="match"`, chamado sobre arquivos como na linha de comando
(`evaluation.py --etype match`). O EM e a dificuldade são lidos da tabela que
ele imprime. O adaptador precisa repetir os dois.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest

from tcc_rag.config import SandboxConfig
from tcc_rag.evaluation import em_oficial, ex_oficial, metrics
from tcc_rag.evaluation.em_oficial import AvaliadorEM, carregar_match
from tcc_rag.evaluation.ex_oficial import AvaliadorEX
from tcc_rag.evaluation.sandbox import Sandbox

TABELA = {
    "db_id": "escola",
    "table_names_original": ["aluno", "avaliacao"],
    "column_names_original": [
        [-1, "*"], [0, "id"], [0, "nome"], [0, "nota"], [0, "turma"],
        [1, "id"], [1, "aluno_id"], [1, "valor"],
    ],
    "column_types": ["text", "number", "text", "number", "text", "number", "number", "number"],
    "primary_keys": [1, 5],
    "foreign_keys": [[6, 1]],
}


@pytest.fixture
def banco(tmp_path: Path) -> Path:
    caminho = tmp_path / "database" / "escola" / "escola.sqlite"
    caminho.parent.mkdir(parents=True)
    conn = sqlite3.connect(caminho)
    conn.executescript(
        """
        CREATE TABLE aluno (id INTEGER PRIMARY KEY, nome TEXT, nota REAL, turma TEXT);
        INSERT INTO aluno VALUES (1, 'Ana', 9.0, 'A'), (2, 'Bruno', 7.5, 'B'),
            (3, 'Celia', 8.25, 'A'), (4, 'Davi', 7.5, 'B'), (5, 'Eva', 6.0, 'C');
        CREATE TABLE avaliacao (id INTEGER PRIMARY KEY, aluno_id INTEGER, valor REAL,
            FOREIGN KEY (aluno_id) REFERENCES aluno(id));
        INSERT INTO avaliacao VALUES (1, 1, 9.5), (2, 1, 8.0), (3, 3, 7.0), (4, 4, 6.5);
        """
    )
    conn.commit()
    conn.close()
    return caminho


def _oficial(banco: Path, tmp_path: Path, pred: str, gold: str) -> tuple[bool, str]:
    """`evaluation.py --etype match` sobre um único item: (EM, dificuldade)."""
    evaluation, _ = carregar_match()
    gold_arq, pred_arq = tmp_path / "gold.txt", tmp_path / "pred.txt"
    gold_arq.write_text(f"{gold}\tescola\n", encoding="utf-8")
    pred_arq.write_text(f"{pred}\n", encoding="utf-8")
    kmaps = {"escola": evaluation.build_foreign_key_map(TABELA)}
    saida = io.StringIO()
    with contextlib.redirect_stdout(saida):
        evaluation.evaluate(str(gold_arq), str(pred_arq), str(banco.parent.parent), "match",
                            kmaps, False, False, False)
    linhas = saida.getvalue().splitlines()
    contagem = next(l for l in linhas if l.startswith("count")).split()[1:]
    exato = next(l for l in linhas if l.startswith("exact match")).split()[2:]
    nivel = [n for n, c in zip(em_oficial.NIVEIS, contagem) if int(c) == 1]
    assert len(nivel) == 1 and int(contagem[4]) == 1
    return float(exato[4]) == 1.0, nivel[0]


CASOS = [
    # (caso, predição, referência, EM esperado)
    ("literal_texto", "SELECT nome FROM aluno WHERE turma = 'B'",
     "SELECT nome FROM aluno WHERE turma = 'A'", True),
    ("literal_numero", "SELECT nome FROM aluno WHERE nota > 9",
     "SELECT nome FROM aluno WHERE nota > 7", True),
    ("limit_valor", "SELECT nome FROM aluno ORDER BY nota DESC LIMIT 3",
     "SELECT nome FROM aluno ORDER BY nota DESC LIMIT 1", True),
    ("marcador_value", "SELECT nome FROM aluno WHERE nota > value",
     "SELECT nome FROM aluno WHERE nota > 8", True),
    ("apelido_e_sem_apelido", "SELECT T1.nome FROM aluno AS T1 WHERE T1.nota > 8",
     "SELECT nome FROM aluno WHERE nota > 8", True),
    ("apelidos_trocados",
     "SELECT T2.nome FROM aluno AS T2 JOIN avaliacao AS T1 ON T2.id = T1.aluno_id",
     "SELECT T1.nome FROM aluno AS T1 JOIN avaliacao AS T2 ON T1.id = T2.aluno_id", True),
    ("ordem_select", "SELECT nota, nome FROM aluno", "SELECT nome, nota FROM aluno", True),
    ("ordem_condicoes", "SELECT nome FROM aluno WHERE turma = 'A' AND nota > 8",
     "SELECT nome FROM aluno WHERE nota > 8 AND turma = 'A'", True),
    ("ordem_join",
     "SELECT T2.nome FROM avaliacao AS T1 JOIN aluno AS T2 ON T1.aluno_id = T2.id WHERE T1.valor > 7",
     "SELECT T1.nome FROM aluno AS T1 JOIN avaliacao AS T2 ON T1.id = T2.aluno_id WHERE T2.valor > 7",
     True),
    ("chave_estrangeira",
     "SELECT T2.aluno_id FROM aluno AS T1 JOIN avaliacao AS T2 ON T1.id = T2.aluno_id",
     "SELECT T1.id FROM aluno AS T1 JOIN avaliacao AS T2 ON T1.id = T2.aluno_id", True),
    ("distinct", "SELECT DISTINCT turma FROM aluno", "SELECT turma FROM aluno", True),
    ("maximo_por_subconsulta", "SELECT nome FROM aluno WHERE nota = (SELECT max(nota) FROM aluno)",
     "SELECT nome FROM aluno ORDER BY nota DESC LIMIT 1", False),
    ("operador_equivalente_nos_dados", "SELECT nome FROM aluno WHERE nota >= 8",
     "SELECT nome FROM aluno WHERE nota > 7.9", False),
    ("count_coluna", "SELECT count(id) FROM aluno", "SELECT count(*) FROM aluno", False),
    ("coluna_diferente", "SELECT nota FROM aluno", "SELECT nome FROM aluno", False),
    ("operador_diferente", "SELECT nome FROM aluno WHERE nota < 8",
     "SELECT nome FROM aluno WHERE nota > 8", False),
    ("tabela_a_mais",
     "SELECT T1.nome FROM aluno AS T1 JOIN avaliacao AS T2 ON T1.id = T2.aluno_id",
     "SELECT nome FROM aluno", False),
    ("palavra_errada", "SELEC nome FROM aluno", "SELECT nome FROM aluno", False),
    ("coluna_inexistente", "SELECT idade FROM aluno", "SELECT nome FROM aluno", False),
    ("apelido_sem_as", "SELECT a.nome FROM aluno a", "SELECT nome FROM aluno", False),
]


@pytest.mark.parametrize("caso,pred,gold,esperado", CASOS, ids=[c[0] for c in CASOS])
def test_adaptador_concorda_com_o_oficial(banco, tmp_path, caso, pred, gold, esperado):
    em_ref, nivel_ref = _oficial(banco, tmp_path, pred, gold)
    veredito = AvaliadorEM().avaliar(banco, TABELA, pred, gold)
    assert em_ref is esperado, "o caso não exercita o que diz"
    assert veredito.correto is em_ref
    assert veredito.dificuldade == nivel_ref


@pytest.mark.parametrize("pred", ["SELEC nome FROM aluno", "SELECT idade FROM aluno",
                                  "SELECT a.nome FROM aluno a", ""])
def test_predicao_nao_analisavel_e_em_falso_explicito(banco, pred):
    veredito = AvaliadorEM().avaliar(banco, TABELA, pred, "SELECT nome FROM aluno")
    assert veredito.correto is False
    assert veredito.situacao == em_oficial.PREDICAO_NAO_ANALISAVEL
    assert veredito.erro.startswith("predição:")
    assert veredito.dificuldade == "easy"


def test_situacao_e_componentes_divergentes(banco):
    avaliador = AvaliadorEM()
    igual = avaliador.avaliar(banco, TABELA, "SELECT nota, nome FROM aluno", "SELECT nome, nota FROM aluno")
    assert (igual.situacao, igual.componentes_divergentes, igual.erro) == ("coincide", (), None)
    errado = avaliador.avaliar(banco, TABELA, "SELECT nome FROM aluno WHERE nota < 8",
                               "SELECT nome FROM aluno WHERE nota > 8")
    assert errado.situacao == "difere"
    assert "where" in errado.componentes_divergentes
    assert "select" not in errado.componentes_divergentes


@pytest.mark.parametrize("pred,gold", [
    ("SELECT nome FROM aluno WHERE nota = (SELECT max(nota) FROM aluno)",
     "SELECT nome FROM aluno ORDER BY nota DESC LIMIT 1"),
    ("SELECT nome FROM aluno WHERE nota >= 8", "SELECT nome FROM aluno WHERE nota > 7.9"),
    ("SELECT count(id) FROM aluno", "SELECT count(*) FROM aluno"),
])
def test_consulta_equivalente_com_outra_estrutura_e_em_falso_ex_verdadeiro(banco, pred, gold):
    assert AvaliadorEX(Sandbox(SandboxConfig())).avaliar(banco, pred, gold).correto is True
    assert AvaliadorEM().avaliar(banco, TABELA, pred, gold).correto is False


def test_literal_diferente_e_em_verdadeiro_ex_falso(banco):
    pred, gold = "SELECT nome FROM aluno WHERE turma = 'B'", "SELECT nome FROM aluno WHERE turma = 'A'"
    assert AvaliadorEX(Sandbox(SandboxConfig())).avaliar(banco, pred, gold).correto is False
    assert AvaliadorEM().avaliar(banco, TABELA, pred, gold).correto is True


NIVEIS = [
    ("SELECT count(*) FROM aluno", "easy"),
    ("SELECT nome, nota FROM aluno WHERE turma = 'A'", "medium"),
    ("SELECT nome FROM aluno WHERE nota > (SELECT avg(nota) FROM aluno)", "hard"),
    ("SELECT T1.nome FROM aluno AS T1 JOIN avaliacao AS T2 ON T1.id = T2.aluno_id "
     "WHERE T2.valor > 5 GROUP BY T1.nome ORDER BY count(*) DESC LIMIT 1", "extra"),
]


@pytest.mark.parametrize("gold,esperado", NIVEIS, ids=[n for _, n in NIVEIS])
def test_dificuldade_nos_quatro_niveis(banco, tmp_path, gold, esperado):
    _, nivel_ref = _oficial(banco, tmp_path, gold, gold)
    assert nivel_ref == esperado
    assert AvaliadorEM().dificuldade(banco, gold) == esperado
    assert AvaliadorEM().avaliar(banco, TABELA, gold, gold).dificuldade == esperado


def test_referencia_nao_analisavel_fica_sem_veredito_e_sem_dificuldade(banco, tmp_path):
    gold = "SELECT a.nome FROM aluno a"
    with pytest.raises(Exception):  # a oficial interrompe a avaliação
        _oficial(banco, tmp_path, "SELECT nome FROM aluno", gold)
    veredito = AvaliadorEM().avaliar(banco, TABELA, "SELECT nome FROM aluno", gold)
    assert veredito.correto is None
    assert veredito.situacao == em_oficial.REFERENCIA_NAO_ANALISAVEL
    assert veredito.dificuldade is None
    assert AvaliadorEM().dificuldade(banco, gold) is None


def test_medida_propria_continua_separada(banco):
    """A forma canônica própria não é o EM oficial, e os dois discordam."""
    pred, gold = "SELECT nota, nome FROM aluno", "SELECT nome, nota FROM aluno"
    assert AvaliadorEM().avaliar(banco, TABELA, pred, gold).correto is True
    assert metrics.em_canonico_proprio(pred, gold) is False


def test_avaliacao_nao_altera_o_banco(banco):
    antes = hashlib.sha256(banco.read_bytes()).hexdigest()
    for _, pred, gold, _ in CASOS:
        AvaliadorEM().avaliar(banco, TABELA, pred, gold)
    assert hashlib.sha256(banco.read_bytes()).hexdigest() == antes


def test_banco_inexistente_falha_sem_criar_arquivo(tmp_path):
    caminho = tmp_path / "nao_existe.sqlite"
    with pytest.raises(FileNotFoundError):
        AvaliadorEM().avaliar(caminho, TABELA, "SELECT 1", "SELECT nome FROM aluno")
    assert not caminho.exists()


@pytest.mark.parametrize("arquivo", ["evaluation.py", "process_sql.py"])
def test_copia_alterada_do_avaliador_de_em_e_recusada(tmp_path, arquivo):
    pasta = tmp_path / "copia"
    shutil.copytree(ex_oficial._PASTA, pasta)
    with (pasta / arquivo).open("a", encoding="utf-8") as fh:
        fh.write("\nDISABLE_VALUE = False\n")
    with pytest.raises(RuntimeError, match="alterada"):
        carregar_match(pasta)


def test_carga_nao_deixa_nomes_genericos_em_sys_modules():
    carregar_match()
    assert "process_sql" not in sys.modules
    assert "exec_eval" not in sys.modules


def test_sem_punkt_tab_a_avaliacao_se_recusa(monkeypatch):
    import nltk

    def ausente(recurso, *args, **kwargs):
        raise LookupError(recurso)

    monkeypatch.setattr(nltk.data, "find", ausente)
    with pytest.raises(RuntimeError, match="punkt_tab"):
        em_oficial._exigir_tokenizador()


def test_identificador_registra_versao_e_regras():
    import nltk

    ident = AvaliadorEM().identificador
    assert ex_oficial.COMMIT[:7] in ident and f"nltk={nltk.__version__}" in ident
    assert "sem valores" in ident and "sem DISTINCT" in ident
