#!/bin/bash
# Dados do experimento na pasta apontada por /content/tcc (idempotente):
#  - Spider 1.0 oficial (https://yale-lily.github.io/spider, CC BY-SA 4.0), conferido por SHA-256
#    contra scripts/colab/spider.sha256 (tables.json, train_spider.json, dev.json e os 23 bancos
#    com volume para o recorte);
#  - perguntas em PT-BR: Marchanjo/spider-pt (Jose e Cozman, 2021; CC BY-SA 4.0) na revisao
#    d9cb7bf3, sem alteracao, convertidas por scripts/gerar_questions_pt.py.
set -eu
cd /content/TCC2
D=/content/tcc/data
if [ ! -f $D/spider/tables.json ]; then
  mkdir -p /content/spider_dl && cd /content/spider_dl
  [ -f spider.zip ] || gdown -q --id 1TqleXec_OykOYFREKKtschzY29dUcVAQ -O spider.zip
  rm -rf x && unzip -q spider.zip -d x
  (cd x/spider && sha256sum --quiet -c /content/TCC2/scripts/colab/spider.sha256)
  mkdir -p $D/spider/database
  cp x/spider/tables.json x/spider/train_spider.json x/spider/dev.json $D/spider/
  grep -o "database/[^/]*" /content/TCC2/scripts/colab/spider.sha256 | sort -u | while read b; do
    cp -r "x/spider/$b" $D/spider/database/
  done
  cd /content/TCC2
fi
(cd $D/spider && sha256sum --quiet -c /content/TCC2/scripts/colab/spider.sha256) && echo "Spider conferido: 26 arquivos"
if [ ! -f $D/spider_pt/questions_pt.json ]; then
  hf download Marchanjo/spider-pt --repo-type dataset --revision d9cb7bf3db3f837856d2730cb1bd24fc4adda679 \
     --local-dir /content/marchanjo_spider_pt > /dev/null
  python -I scripts/gerar_questions_pt.py $D/spider /content/marchanjo_spider_pt $D/spider_pt/questions_pt.json > /dev/null
fi
echo "e5584fad25fc3364a3d644e85b39f8a1795472aa13111fa4e62a2f3bc4209d0e  $D/spider_pt/questions_pt.json" | sha256sum --quiet -c - && echo "questions_pt.json conferido"
