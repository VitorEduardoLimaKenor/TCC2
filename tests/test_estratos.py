"""Regra pré-registrada do estrato de literal traduzido (LT/SL)."""

import pytest

from tcc_rag.data.estratos import estrato, literais_de_texto, literais_traduzidos


@pytest.mark.parametrize("sql,esperado", [
    ("SELECT a FROM t WHERE n = 'Statistics'", ["Statistics"]),
    ('SELECT a FROM t WHERE n = "Fog" OR n = "Rain"', ["Fog", "Rain"]),
    ("SELECT a FROM t WHERE n LIKE '%Mar%'", ["Mar"]),
    ("SELECT a FROM t WHERE n = 'O''Brien'", ["O''Brien"]),
    ("SELECT a FROM t WHERE y = '2003' OR z = '-1.5' OR w = '%10%'", []),  # numéricos
    ("SELECT a FROM t WHERE n = 'F' OR m = '%a%'", []),  # menos de 2 caracteres
    ("SELECT a FROM t WHERE n = 3", []),
])
def test_literais_de_texto(sql, esperado):
    assert literais_de_texto(sql) == esperado


def test_traduzido_entra_em_lt():
    sql = "SELECT title FROM course WHERE dept_name = 'Statistics'"
    assert estrato(sql, "Which courses are in Statistics?", "Quais cursos são de Estatística?") == "LT"


def test_preservado_fica_em_sl_sem_distinguir_maiusculas():
    sql = "SELECT a FROM t WHERE n = 'Smith Hall'"
    assert estrato(sql, "students in Smith Hall", "alunos no SMITH HALL") == "SL"


def test_ausente_da_pergunta_em_ingles_fica_em_sl():
    sql = "SELECT a FROM t WHERE allergytype = 'food'"
    assert literais_traduzidos(sql, "What are the foodstuffs?", "Quais itens?") == []
    assert estrato(sql, "Show the allergies.", "Mostre as alergias.") == "SL"


def test_fronteira_de_palavra():
    sql = "SELECT a FROM t WHERE c = 'Mar'"
    assert estrato(sql, "released in Mar", "lançado em março") == "LT"
    # "Mar" dentro de "Marcos" não é a palavra: na PT não conta como presente.
    assert estrato(sql, "released in Mar", "lançado por Marcos") == "LT"
    # E dentro de "March" não conta como presente na EN.
    assert estrato(sql, "released in March", "lançado em março") == "SL"


def test_basta_um_literal_traduzido():
    sql = "SELECT a FROM t WHERE c = 'Fog' AND d = 'Rain'"
    assert literais_traduzidos(sql, "neither Fog nor Rain", "nem Fog nem chuva") == ["Rain"]
