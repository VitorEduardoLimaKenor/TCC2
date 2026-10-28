# Avaliador oficial do Spider (EX, EM e dificuldade) — cópia fixada

Repositório: <https://github.com/taoyds/test-suite-sql-eval>
Commit: `e97acc546ecbee8fa27fa8dbf025ef61493a876c` (02/08/2022, "Create LICENSE";
última mudança de código em `48cb78e`, 11/04/2021).
Licença: Apache-2.0 (`LICENSE`, copiado do mesmo commit).
Referência: Zhong, Yu e Klein (2020), *Semantic Evaluation for Text-to-SQL with
Distilled Test Suites*, EMNLP. Segundo o README do repositório, a acurácia por
*test suite* "is now the official metric of Spider, SParC, and CoSQL" desde
outubro de 2020.

| Arquivo | Blob git (upstream) | SHA-256 (bytes LF) |
|---|---|---|
| `exec_eval.py` | `9bc34ba75eff4446223248fefb99fc429b7d23e7` | `29d034db28904490c28037537a14fbb0150b6e86cef0049076c0511d6b6b77f7` |
| `parse.py` | `5271f5f86c9de2e0171cad8c690d7a885752f7e0` | `ef04211a6e1c1e142571157f5c1999613e3451084c044083b2de1977f1f622c5` |
| `evaluation.py` | `fae509c6c3afe60635e0ae9971c2204be7637005` | `7401e4014a8955376a7919c06903a7f0ab403c99e89f94204cd8f4c8e32ae779` |
| `process_sql.py` | `fd93ea287dd80f7518c995767917a4cc1012d626` | `927fc564f7a8e34f09f009a2f5564a83fdf95226440dde84c87871fd65fe55a1` |
| `LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

Os arquivos foram extraídos com `git cat-file -p <commit>:<arquivo>` e não foram
modificados: `exec_eval.py` e `parse.py` na E02 (27/10/2026), `evaluation.py` e
`process_sql.py` na E03 (28/10/2026), do mesmo commit. Os dois últimos vêm do
commit `82793e5` (20/10/2020) e não mudaram depois dele. `ex_oficial.py` confere
o blob dos quatro antes de carregar qualquer um e recusa uma cópia alterada. O
`.gitattributes` do projeto desliga a conversão de fim de linha nesta pasta.

## O que é usado

De `exec_eval.py`: `postprocess`, `replace_cur_year` e `result_eq`; de
`parse.py`: `remove_distinct` (que depende de `sqlparse`, fixado no
`pyproject.toml`). As transformações que o `evaluation.py` oficial aplica antes de
chamar `eval_exec_match` (`strip()` das duas consultas e `replace("value", "1")`
na predição) são reproduzidas em `ex_oficial.py`, com indicação das linhas.

## O que não é usado, e por quê

A execução (`exec_on_db`) abre o banco para escrita e envolve uma chamada
síncrona em `asyncio.wait_for`, de modo que o tempo limite de 60 s não interrompe
a consulta. O projeto executa as mesmas consultas no `Sandbox`: processo
separado, banco somente leitura e tempo limite imposto de fora. Para consultas de
leitura que terminam no prazo, o veredito é o mesmo; as diferenças possíveis
estão descritas em `ex_oficial.py`.

## EM e dificuldade (E03)

De `evaluation.py`: `Evaluator.eval_exact_match` (com `eval_partial_match`),
`Evaluator.eval_hardness`, `build_valid_col_units`, `rebuild_sql_val`,
`rebuild_sql_col` e `build_foreign_key_map`; de `process_sql.py`: `Schema`,
`get_schema` e `get_sql`. É o que `evaluation.py --etype match` calcula, a
*exact set match accuracy* original, que o README do repositório diz continuar
reportando "as a reference" ao lado da *test suite*. `em_oficial.py` repete os
passos das linhas 568–628 do `evaluate` oficial (`strip()`, "value" → "1" na
predição, consulta vazia das linhas 589–606 quando a predição não é analisável)
e não reimplementa nenhuma regra de comparação.

`process_sql.py` tokeniza com `nltk.word_tokenize`. O repositório não fixa a
versão do nltk; o projeto fixa `nltk==3.9.4` no `pyproject.toml`, que exige o
modelo `punkt_tab`. O `evaluate` oficial não é chamado pelo pipeline, porque lê
arquivos com a codificação padrão do sistema, imprime em vez de devolver e
interrompe a avaliação inteira quando uma referência não é analisável. Os
testes o chamam diretamente, como referência.

Não são usados: `evaluate` (só nos testes e na verificação), `isValidSQL`, as
funções de impressão e `build_foreign_key_map_from_json`, que abre o
`tables.json` sem fixar a codificação; o projeto lê o arquivo em UTF-8 e passa
cada entrada a `build_foreign_key_map`.
