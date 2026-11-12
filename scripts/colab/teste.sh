#!/bin/bash
# E16: teste confirmatorio (471 perguntas, 5 condicoes, 2 idiomas).
# So depois do manifesto de congelamento (docs/congelamento-*.md). Retomavel.
# Exige arvore limpa: todas as linhas do teste saem de uma unica revisao do codigo.
set -eu -o pipefail
cd /content/TCC2 || exit 1
CFG=configs/confirmatorio.yaml
F='it/s\]|s/it\]|Warning|warn'
if [ -n "$(git status --porcelain)" ]; then echo "arvore suja: abortado"; exit 1; fi
echo "codigo: $(git rev-parse HEAD)"
for L in pt en; do
  echo "== teste $L inicio $(date -u +%FT%TZ)"
  tcc-rag run --config $CFG --split test --langs $L 2>&1 | grep -v -E "$F"
  echo "== teste $L fim $(date -u +%FT%TZ)"
done
tcc-rag evaluate --config $CFG --split test 2>&1 | grep -v -E "$F"
echo "== FIM GERACAO DO TESTE $(date -u +%FT%TZ)"
