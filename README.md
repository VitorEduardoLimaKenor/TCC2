# Recuperação de demonstrações orientada à estrutura para Text-to-SQL

Código do experimento da monografia. Compara cinco condições de seleção de
demonstrações em português brasileiro e em inglês, sob esquema, número de
exemplos e orçamento de contexto constantes.

| Condição | Demonstrações no *prompt* | Papel |
|---|---|---|
| C0 | nenhuma | ponto de partida do modelo no ambiente |
| C1 | *k* exemplos fixos do banco | separa "ver o domínio" de "receber exemplo pertinente" |
| C2 | *top-k* por similaridade da pergunta original | **comparação principal** |
| C3 | *top-k* por similaridade da pergunta delexicalizada | isola a primeira etapa |
| C4 | recuperação em duas etapas | **método proposto** |

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
pytest -q
```

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
- o sandbox rejeita escrita e instrução múltipla, e não altera o banco;
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
    metrics.py        EX, EM, acerto estrutural, agregados
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
