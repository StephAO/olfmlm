#!/bin/bash

RANK=0
WORLD_SIZE=1

python3 -m sentence-encoders.pretrain_bert \
    --batch-size 32 \
    --tokenizer-type BertWordPieceTokenizer \
    --cache-dir cache_dir \
    --tokenizer-model-type bert-base-uncased \
    --vocab-size 30522 \
    --train-data wikipedia \
    --presplit-sentences \
    --text-key text \
    --split 1000,1,1 \
    --lazy-loader \
    --max-preds-per-seq 80 \
    --seq-length 128 \
    --train-iters 250000 \
    --lr 0.0001 \
    --lr-decay-style linear \
    --warmup .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --num-workers 2 \
    --epochs 1 \
    --bert-config-file /h/stephaneao/trained_berts/config_file.json \
    --save /scratch/gobi1/stephaneao/trained_berts/corrupt/ \
    --model-type 'corrupt' \
    --lr-decay-iters 225000 \
    --track-results True