# Resultados da rodada confirmatória (14–15/11/2026)

Registros que sustentam os Capítulos 5 e 6 da monografia. Código da geração:
revisão `ff4abc8`; configuração `configs/confirmatorio.yaml` (`config_hash`
`a99f7394fedc945b`); partição `948225fbfa08dab4`.

| Arquivo | Conteúdo |
|---|---|
| `artifacts/manifest.json` | Perguntas do recorte (PT-BR e EN), consulta de referência, família e partição de cada uma |
| `artifacts/estratos_literais.json` | Estrato de literal traduzido de cada pergunta |
| `artifacts/leakage_report.json` | Auditoria de vazamento entre as partições |
| `artifacts/congelamento.json` | Registro de congelamento, gravado antes da abertura do teste |
| `artifacts/runs-v2/*-val-*.jsonl` | Registros por pergunta da validação (275 × 5 condições × 2 idiomas) |
| `artifacts/runs-v2/*-test-*.jsonl` | Registros por pergunta do teste (471 × 5 condições × 2 idiomas) |
| `artifacts/custo/` | Passagem de latência com uma pergunta por vez (500 medições) |
| `artifacts/cache/rascunhos-v2.jsonl` | Consultas preliminares da condição C4 |
| `artifacts/analise/analise-val-*.json` | Análise da validação |
| `analise-test-a99f7394fedc945b.json` | Análise do teste, de onde saem as tabelas e figuras do Capítulo 5 |
| `SHA256SUMS.txt` | Somas de verificação |

Para refazer a análise do teste e as tabelas (os bancos do Spider ficam em `data/spider`):

```bash
python scripts/analise_local.py resultados/confirmatorio/artifacts data/spider test analise-test.json
python scripts/tabelas_cap5.py resultados/confirmatorio/artifacts/analise/analise-val-a99f7394fedc945b.json analise-test.json <saida>
```

## Origem dos dados e licença

As perguntas em inglês, as consultas de referência e os bancos são do Spider 1.0
(Yu et al., 2018); as perguntas em português brasileiro são da tradução de José e
Cozman (2021), `Marchanjo/spider-pt`. Ambos são distribuídos sob CC BY-SA 4.0, e os
arquivos desta pasta, que os reproduzem em parte, seguem a mesma licença.
