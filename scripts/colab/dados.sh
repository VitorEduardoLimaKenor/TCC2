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
  # Dois arquivos, baixados diretamente na revisao fixada (com novas tentativas: o Hub
  # limita requisicoes anonimas) e conferidos por SHA-256 antes do uso.
  mkdir -p /content/marchanjo_spider_pt
  R=https://huggingface.co/datasets/Marchanjo/spider-pt/resolve/d9cb7bf3db3f837856d2730cb1bd24fc4adda679
  for f in train_spider.json dev.json; do
    curl -sfL --retry 10 --retry-delay 20 --retry-all-errors -o /content/marchanjo_spider_pt/$f $R/$f
  done
  (cd /content/marchanjo_spider_pt && sha256sum --quiet -c - <<'SHA'
642b07c284e3da9a2282e1db71801fefe5276730327c9a26349aa7b9e2c03088  train_spider.json
673721b6d40d66e9782f02f9fdfc868c5bf7c6bf8710a99ed732ae978b3fcd7c  dev.json
SHA
  )
  python -I scripts/gerar_questions_pt.py $D/spider /content/marchanjo_spider_pt $D/spider_pt/questions_pt.json > /dev/null
fi
echo "e5584fad25fc3364a3d644e85b39f8a1795472aa13111fa4e62a2f3bc4209d0e  $D/spider_pt/questions_pt.json" | sha256sum --quiet -c - && echo "questions_pt.json conferido"
