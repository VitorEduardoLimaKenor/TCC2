#!/bin/bash
# Rodada completa e retomavel: preparo -> validacao -> congelamento -> teste -> custo e analise.
# Cada passo pula o que ja esta feito. O teste so comeca depois que `tcc-rag congelar` grava o
# registro de congelamento; se o portao recusar, a rodada para e o teste continua selado.
set -u
cd /content/TCC2 || exit 1
CFG=configs/confirmatorio.yaml
bash scripts/colab/preparar.sh   || { echo "PREPARO FALHOU"; exit 1; }
bash scripts/colab/validacao.sh  || { echo "VALIDACAO FALHOU"; exit 1; }
echo "== congelamento $(date -u +%FT%TZ)"
tcc-rag congelar --config $CFG   || { echo "TESTE SELADO: congelamento recusado"; exit 1; }
bash scripts/colab/teste.sh      || { echo "TESTE INTERROMPIDO"; exit 1; }
bash scripts/colab/custo_e_analise.sh || { echo "CUSTO/ANALISE FALHOU"; exit 1; }
echo "== RODADA COMPLETA $(date -u +%FT%TZ)"
