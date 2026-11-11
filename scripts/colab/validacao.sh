#!/bin/bash
# E11: validacao PT/EN completa (275 perguntas, 5 condicoes), configuracao candidata.
# Retomavel: o que ja esta no log e pulado.
set -u
cd /content/TCC2 || exit 1
CFG=configs/confirmatorio.yaml
F='it/s\]|s/it\]|Warning|warn'
echo "codigo: $(git rev-parse HEAD) | sujo: $(git status --porcelain | wc -l)"
for L in pt en; do
  echo "== val $L inicio $(date -u +%FT%TZ)"
  tcc-rag run --config $CFG --split val --langs $L 2>&1 | grep -v -E "$F"
  echo "== val $L fim $(date -u +%FT%TZ)"
done
tcc-rag evaluate --config $CFG --split val 2>&1 | grep -v -E "$F"
tcc-rag analise  --config $CFG --split val --sensibilidade-distinct 2>&1 | grep -v -E "$F"
echo "== FIM VALIDACAO $(date -u +%FT%TZ)"
