#!/bin/bash
# make fp-quant variants on sub
set -e
. ~/fp12-exp/env.sh
cd $E
(cd fp12src && ~/.cargo/bin/cargo test 2>&1 | tail -15) > logs/cargo-test.log
for v in "fp12 fp12" "e4m7 e4m7" "e6m5 e6m5" "fp8 fp8 --scale" "e5m2 e5m2 --scale"; do
  set -- $v; name=$1; fmt=$2; shift 2
  mkdir -p variants/$name
  cp models/base/*.json models/base/merges.txt models/base/LICENSE variants/$name/
  $FQ models/base/model.safetensors variants/$name/model.safetensors --format $fmt "$@" > logs/fpquant-$name.log 2>&1
  echo "== $name"; tail -3 logs/fpquant-$name.log
done
