# Avaliador oficial de execução do Spider — cópia fixada

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
| `LICENSE` | `261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64` | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

Os arquivos foram extraídos com `git cat-file -p <commit>:<arquivo>` e não foram
modificados. `ex_oficial.py` confere o blob de cada um antes de carregá-los e
recusa uma cópia alterada. O `.gitattributes` do projeto desliga a conversão de
fim de linha nesta pasta.

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

O `evaluation.py` (leitura de arquivos, EM e dificuldade) não foi copiado para esta
etapa. EM e dificuldade oficiais pertencem à etapa E03.
