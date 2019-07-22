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
    --train-iters 100000 \
    --lr 0.0001 \
    --lr-decay-style linear \
    --warmup .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --num-workers 2 \
    --epochs 1 \
    --bert-config-file /h/stephaneao/trained_berts/config_file.json \
    --bert-small-config-file /h/stephaneao/sentence-encoders/bert_small_config.json \
    --save /scratch/gobi1/stephaneao/trained_berts/referential_game/ \
    --model-type 'referential_game' \
    --track-results True
