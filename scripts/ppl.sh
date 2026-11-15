#!/bin/bash
# quantize references + perplexity for all 9 on sub
. ~/fp12-exp/env.sh
cd $E
md5sum gguf/*-bf16.gguf > logs/md5.txt
$B/llama-quantize gguf/base-bf16.gguf gguf/base-q6_k.gguf Q6_K > logs/quant-q6_k.log 2>&1
$B/llama-quantize gguf/base-bf16.gguf gguf/base-q4_k_m.gguf Q4_K_M > logs/quant-q4_k_m.log 2>&1
for n in base-bf16 fp12-bf16 e4m7-bf16 e6m5-bf16 fp8-bf16 e5m2-bf16 base-q8_0 base-q6_k base-q4_k_m; do
  $B/llama-perplexity -m gguf/$n.gguf -f data/wiki.test.raw -ngl 99 > logs/ppl-$n.log 2>&1
  echo "$n $(stat -c %s gguf/$n.gguf) $(grep -o 'Final estimate: PPL = .*' logs/ppl-$n.log)"
done
