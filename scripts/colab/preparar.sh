#!/bin/bash
# Preparo do ambiente no Colab (GPU T4), depois de montar o Drive em /content/drive.
# Baixa e confere os dados se ainda nao estiverem na pasta. Idempotente.
# A pasta de dados e artefatos e a indicada em TCC_DRIVE (padrao abaixo); /content/tcc aponta
# para ela. Nunca apontar para a pasta `tcc` de agosto/setembro de 2026, cujo questions_pt.json
# e o provisorio em ingles: `index`, `run` e `custo` recusam esse arquivo.
set -eu
TCC_DRIVE=${TCC_DRIVE:-/content/drive/MyDrive/tcc-confirmatorio}
mkdir -p "$TCC_DRIVE/artifacts"
ln -sfn "$TCC_DRIVE" /content/tcc
cd /content/TCC2 || exit 1
pip install -q -e ".[hf]" bitsandbytes
python -m nltk.downloader punkt_tab > /dev/null
bash scripts/colab/dados.sh
CFG=configs/confirmatorio.yaml
tcc-rag prepare   --config $CFG | tail -3
tcc-rag partition --config $CFG | tail -4      # deve imprimir partition_digest: 948225fbfa08dab4
tcc-rag leakage   --config $CFG | tail -2      # VEREDITO: limpo
tcc-rag estratos  --config $CFG                # pool 204/917, val 46/229, test 80/391
# O indice so e reconstruido se ainda nao existir: reconstruir muda o indice_digest
# e impede a retomada de uma rodada em andamento.
if [ ! -f /content/tcc/artifacts/qdrant-confirmatorio.demos.identidade.json ]; then
  tcc-rag index --config $CFG
fi
