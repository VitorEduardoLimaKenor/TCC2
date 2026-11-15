#!/bin/bash
# Passagem de latencia (lote unitario) numa sessao separada, com o codigo da revisao congelada.
# Pasta local no lugar do Drive: os dados sao baixados e conferidos por SHA, a particao e o
# indice sao reconstruidos e conferidos contra o registro do teste antes de qualquer medicao.
set -eu -o pipefail
REV=ff4abc8ac8bfa6867b21c9148e37719854053507
[ -d /content/TCC2/.git ] || git clone -q https://github.com/VitorEduardoLimaKenor/TCC2.git /content/TCC2
cd /content/TCC2 && git fetch -q origin && git checkout -q --detach $REV
echo "codigo: $(git rev-parse HEAD) $(git status --porcelain | wc -l) alteracoes"
TCC_DRIVE=/content/tcc-local bash scripts/colab/preparar.sh
A=/content/tcc/artifacts
echo "== conferencia $(date -u +%FT%TZ)"
sha256sum $A/manifest.json $A/estratos_literais.json
echo "4c96a86f5670b1d974f6f7e5fd2e0a4c1b03e56404dadf68376d7becc6126892  $A/manifest.json" | sha256sum --quiet -c - && echo "manifesto identico ao do teste"
echo "f78bf3630ca3e03af9515f4a62dc4b129d8b4f02dbb98575e8cfecf0441bc1bc  $A/estratos_literais.json" | sha256sum --quiet -c - && echo "estratos identicos aos do teste"
D=$(python -c "import json;print(json.load(open('$A/qdrant-confirmatorio.demos.identidade.json'))['digest'])")
echo "indice: $D"
[ "$D" = "8eeffab149bcfc62" ] || { echo "INDICE DIFERENTE DO DO TESTE: parado antes de medir"; exit 3; }
echo "== custo $(date -u +%FT%TZ)"
tcc-rag custo --config configs/confirmatorio.yaml --split test --limit 100 --langs pt 2>&1 | grep --line-buffered -v -E 'it/s\]|s/it\]|Warning|warn'
echo "== FIM LATENCIA $(date -u +%FT%TZ)"
