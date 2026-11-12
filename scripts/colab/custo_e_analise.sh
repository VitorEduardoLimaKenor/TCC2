#!/bin/bash
# Depois do teste: passagem de custo (lote unitario, 100 perguntas do teste em PT,
# amostra estratificada por banco) e analise completa do teste.
set -eu -o pipefail
cd /content/TCC2 || exit 1
CFG=configs/confirmatorio.yaml
F='it/s\]|s/it\]|Warning|warn'
echo "codigo: $(git rev-parse HEAD)"
tcc-rag custo   --config $CFG --split test --limit 100 --langs pt 2>&1 | grep -v -E "$F"
tcc-rag analise --config $CFG --split test --sensibilidade-distinct 2>&1 | grep -v -E "$F"
echo "== FIM CUSTO E ANALISE $(date -u +%FT%TZ)"
