# Recuperação de demonstrações orientada à estrutura para Text-to-SQL

Código do experimento da monografia. Compara cinco condições de seleção de
demonstrações em português brasileiro e em inglês, sob esquema, número de
exemplos e orçamento de contexto constantes.

| Condição | Demonstrações no *prompt* | Papel |
|---|---|---|
| C0 | nenhuma | ponto de partida do modelo no ambiente |
| C1 | *k* exemplos fixos do banco | separa "ver o domínio" de "receber exemplo pertinente" |
| C2 | *top-k* por similaridade da pergunta original | **comparação principal** |
| C3 | *top-k* por similaridade da pergunta delexicalizada | método da comparação primária após a validação |
| C4 | recuperação em duas etapas | análise de componente; preserva H1 original |

## Instalação

Localmente, prefira **Python 3.12**: é onde a pilha de *machine learning* tem
*wheels* para tudo sem compilar nada. O pacote não impõe teto de versão, porque
o Colab e o Kaggle atualizam o Python quando querem e o experimento precisa
rodar lá — no Colab, use o Python que vier, que o torch já está instalado.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[hf,dev]"      # desenvolvimento local
pip install -e ".[vllm,dev]"    # GPU do experimento confirmatório
python -m nltk.downloader punkt_tab   # tokenizador do EM oficial
pytest -q
```

O EM e a dificuldade oficiais usam o analisador do Spider, que tokeniza com
`nltk.word_tokenize`; sem o modelo `punkt_tab`, a avaliação se recusa a começar.
No Colab, rode o mesmo `python -m nltk.downloader punkt_tab` depois do
`pip install`.

## Dados

Dois insumos, ambos fora do repositório:

```
data/spider/
  tables.json
  train_spider.json
  dev.json
  database/<db_id>/<db_id>.sqlite
data/spider_pt/
  questions_pt.json
```

`questions_pt.json` alinha cada pergunta traduzida ao exemplo original, por
identificador `spider_<arquivo>_<índice>`. Dois formatos são aceitos:

```json
{ "spider_train_spider_0": "Quantos cantores existem?" }
```

```json
[ { "id": "spider_train_spider_0", "question_pt": "Quantos cantores existem?" } ]
```

Exemplos sem tradução são descartados e contados. A avaliação é pareada: um
item sem par não entra.

## Uso

```bash
tcc-rag prepare    --config configs/dev-local.yaml   # conta famílias por banco
tcc-rag partition  --config configs/dev-local.yaml   # aloca e audita
tcc-rag leakage    --config configs/dev-local.yaml   # porta obrigatória
tcc-rag index      --config configs/dev-local.yaml   # popula o Qdrant
tcc-rag run        --config configs/dev-local.yaml --split val
tcc-rag evaluate   --config configs/dev-local.yaml
tcc-rag stats      --config configs/dev-local.yaml
tcc-rag tables     --config configs/dev-local.yaml
```

`run` é retomável e idempotente: o que já está no log é pulado. Isso é o que
permite atravessar quedas de sessão no Colab e no Kaggle.

Para ensaiar o encadeamento inteiro sem GPU, troque `generation.backend` para
`dummy`. Os números não significam nada, mas todo o caminho é exercitado.

### Seleção dos logs para relatórios

`evaluate`, `stats` e `tables` selecionam o hash do arquivo `--config` por
padrão. Nunca escolhem o último arquivo por ordem alfabética. Se houver mais
de um log compatível, especifique `--run-id`. É possível combinar logs de
condições separadas, desde que tenham o mesmo hash, partição e split, sem
duplicar observações:

```bash
tcc-rag evaluate --config configs/colab.yaml --run-id rodada-c2 rodada-c3 --split val --partition-digest 948225fbfa08dab4
tcc-rag stats --config configs/colab.yaml --run-id rodada-c2 rodada-c3 --split val --partition-digest 948225fbfa08dab4
```

Os nomes acima são exemplos: use os IDs gravados nos arquivos reais.
`--config-hash HASH` permite selecionar uma configuração histórica ou um
ponto da varredura com hash diferente do YAML atual. Isso não permite misturar
hashes. Linhas inválidas, resultados ausentes e observações duplicadas geram
erro; `stats` e `tables` também recusam conjuntos de itens incompletamente
pareados. Logs antigos sem `split` são recusados: sua proveniência deve ser
recuperada do manifesto original em uma migração auditada, sem inferir o split
apenas pelo nome do arquivo e sem alterar silenciosamente os originais.

Cada seleção também exige **um único avaliador** de EX e de EM. Os logs
anteriores à E02 (sem `ex_status` nem `ex_avaliador`) continuam legíveis, mas
só isolados: aparecem como avaliador "próprio anterior à E02" e não se combinam
com logs novos. Para lê-los, aponte o diretório deles:

```bash
tcc-rag evaluate --config configs/colab.yaml --runs-dir /content/drive/MyDrive/tcc/artifacts/runs --run-id sweep-C2-k4-l95-n25-92419f77
```

### Logs, retomada e caches (E04)

- **Diretório:** os logs novos vão para `<work_dir>/runs-v2/`. Os de `runs/`
  são históricos (texto provisório em inglês) e não são retomados.
- **Nome:** o `run_id` padrão inclui o digest do conteúdo das perguntas
  (`run`: `{nome}-{split}-{config_hash}-{conteúdo}`; `sweep`:
  `sweep-…-{amostra}-{conteúdo}`). Trocar o texto gera outro log.
- **Retomada:** só pula linhas com a mesma identidade (`formato_log`,
  `config_hash`, `partition_digest`, `split`, avaliadores de EX e EM e
  `indice_digest`) e com o mesmo `item_sha256` (pergunta, delexicalização,
  referência e esquema). Qualquer diferença gera erro, nunca mistura. Linhas de
  outra revisão do código geram só aviso, e cada linha registra a sua.
- **Queda no meio da escrita:** a linha final incompleta vai para
  `<log>.incidente-<data>.txt` ao reabrir.
- **Rascunhos de C4:** ficam em `cache/rascunhos-v2.jsonl`, sob o hash da
  identidade completa (configuração de geração, versões das bibliotecas,
  idioma, *template*, pergunta, DDL e prompt). Os `drafts_*.json` antigos vão
  para `cache/quarentena/` e não são usados.
- **Vetores:** o cache usa `emb2_*` e inclui na chave o prefixo E5, a
  normalização, o dispositivo, o lote e as versões das bibliotecas.
- **Índice:** `index` grava a identidade do índice (pool e vetorizador) ao lado
  do diretório do Qdrant. `run` e `sweep` recusam um índice sem ela ou
  construído com outro pool; nesse caso, rode `tcc-rag index`.

`evaluate` grava `summary.json` e `summary_selection.json`; `stats` grava
`stats.json`, incluindo a seleção e as estimativas. Os valores-p são nominais,
sem correção de multiplicidade: o plano inferencial precisa ser formalizado
antes do teste. C3×C2 é a comparação primária; C4×C2 preserva H1; o contraste
linguístico de H2 continua usando C4 e C0. A célula de inspeção do notebook
reutiliza a seleção do último `evaluate` bem-sucedido.

## Acurácia de execução

A EX é a do avaliador oficial do Spider, o `exec_eval.py` do
[test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval) no commit
`e97acc5`, copiado sem alteração em `evaluation/_spider_oficial/` e conferido
pelo blob git antes de cada uso. A comparação segue as regras oficiais:
multiconjunto de linhas, permutação de colunas, ordem apenas quando a referência
contém "order by" e remoção de DISTINCT nas duas consultas (`keep_distinct=False`,
padrão oficial). A predição também tem "value" trocado por "1", como no
`evaluation.py` oficial. Só a execução é do projeto: banco original, somente
leitura, processo separado e tempo limite de 15 s.

Cada linha do log traz `ex_status`:

| `ex_status` | Significado | `ex_correct` |
|---|---|---|
| `coincide` | resultados equivalentes (inclusive dois resultados vazios, `ex_ambos_vazios`) | `true` |
| `difere` | resultados diferentes; com mais linhas que a referência, a leitura para em referência + 1 | `false` |
| `erro_predicao` | a predição falha, tenta escrever ou é vazia | `false` |
| `tempo_predicao` | a predição excede o tempo limite | `false` |
| `referencia_invalida` | a referência falha ou excede o tempo; o item sai dos denominadores e é contado à parte | `null` |

A acurácia por *test suite* não é calculada: as suítes destiladas cobrem apenas
os bancos do desenvolvimento do Spider, e só `car_1` e `world_1` estão no recorte.

## Correspondência exata e dificuldade

O EM é a *exact set match* original do Spider: `Evaluator.eval_exact_match` do
`evaluation.py`, com o analisador `process_sql.py`, do mesmo commit e copiados
da mesma forma. As duas consultas são comparadas cláusula a cláusula, como
conjuntos, sem valores literais, sem DISTINCT e com as chaves estrangeiras do
`tables.json` unificadas. A dificuldade é `Evaluator.eval_hardness` sobre a
referência: `easy`, `medium`, `hard` ou `extra`.

| `em_status` | Significado | `em_oficial` |
|---|---|---|
| `coincide` | as duas consultas coincidem em todos os componentes | `true` |
| `difere` | algum componente difere (`em_componentes_divergentes` diz qual) | `false` |
| `predicao_nao_analisavel` | o analisador oficial recusa a predição, comparada então com a consulta vazia oficial | `false` |
| `erro_comparacao` | a comparação falha depois de analisadas as duas consultas | `false` |
| `referencia_nao_analisavel` | o analisador recusa a referência; o item sai do denominador do EM e fica sem dificuldade oficial | `null` |

As medidas próprias continuam no log, com nomes que não se confundem com as
oficiais: `em_canonico_proprio`, a igualdade das formas canônicas que definem
as famílias da partição, e `dificuldade_aprox`, a faixa estimada pela
assinatura estrutural (o campo `difficulty` do manifesto). Logs anteriores à
E03 as gravaram como `em_correct` e `difficulty` e não têm EM oficial;
`evaluate` mostra "--" no EM desses logs e recusa misturá-los com logs novos.

## Ordem de trabalho

O avaliador vem antes do RAG. Enquanto `tcc-rag run --conditions C0` não
produzir log completo com EX, latência e tokens, não vale construir o
recuperador — qualquer melhoria seria invisível.

1. `prepare` até a porta de decisão: há bancos com famílias suficientes?
2. `partition` + `leakage` até o relatório sair limpo.
3. `run --conditions C0` e a rodada de calibração contra um número publicado.
4. `index`, depois C1, C2, C3.
5. C4.
6. Validação: varredura de `k`, `n_candidates`, `lambda_question`. Congelar.
7. Teste confirmatório, sem reajuste.

## O que os testes protegem

Não são testes de acurácia. São as invariantes que, se quebrarem, invalidam o
resultado sem dar sinal:

- família nunca cruza partição;
- demonstração recuperada nunca vem de fora do pool nem de outro idioma;
- o gabarito de um item de teste nunca aparece entre as demonstrações;
- entre duas condições, o *prompt* difere apenas no bloco de demonstrações;
- a delexicalização não consulta o SQL de referência;
- o sandbox rejeita escrita, ATTACH e PRAGMA, e não altera o banco;
- a EX do projeto coincide com a função oficial `eval_exec_match` nos casos
  sintéticos, e a cópia do avaliador oficial é idêntica à do commit fixado;
- reexecutar não duplica linha no log.

## Estrutura

```
src/tcc_rag/
  config.py           configuração + hash que identifica a execução
  schemas.py          tipos; o gabarito só existe em TestItem
  pipeline.py         manifesto -> demonstrações e itens de teste
  runner.py           orquestração, em lote e retomável
  cli.py              linha de comando
  data/
    spider.py         carga, DDL determinístico, faixa de dificuldade
    normalize.py      forma canônica e família
    delex.py          delexicalização (superficial, por exigência do desenho)
    partition.py      alocação por família
    leakage.py        auditoria; assert_clean levanta exceção
  structure/
    signature.py      assinatura estrutural, similaridade, compatibilidade
  retrieval/
    embedder.py       vetorização com cache em disco
    index.py          Qdrant local ou servidor, com filtro por banco e idioma
    selectors.py      as cinco condições, num único despacho
  generation/
    prompt.py         montador único; sha256 do prompt vai para o log
    llm.py            backends dummy / transformers / vLLM
  evaluation/
    sandbox.py        execução isolada, somente leitura, com timeout
    ex_oficial.py     EX pelo avaliador oficial do Spider (cópia em _spider_oficial/)
    metrics.py        EM, acerto estrutural, agregados
    stats.py          McNemar exato, bootstrap pareado, diferenças em diferenças
  report/
    tables.py         tabelas LaTeX do Capítulo 5
```

## No Colab e no Kaggle

Não escreva o experimento dentro do *notebook*. O *notebook* é isto:

```python
!git clone https://github.com/<usuario>/<repo>.git
%cd <repo>
!pip install -q -e ".[vllm]"
from google.colab import drive; drive.mount('/content/drive')
!tcc-rag run --config configs/frozen.yaml --split test
```

O Qdrant não precisa de Docker lá: `qdrant.mode: local` guarda o índice num
diretório. Aponte-o para o Drive e ele sobrevive entre sessões.
