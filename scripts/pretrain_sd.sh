#!/bin/bash

RANK=0
WORLD_SIZE=1

python3 sentence_encoders/pretrain_bert.py \
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
    --train-iters 50000 \
    --lr 0.0001 \
    --lr-decay-style linear \
    --warmup .01 \
    --weight-decay 1e-2 \
    --clip-grad 1.0 \
    --num-workers 0 \
    --epochs 2 \
    --bert-config-file /h/stephaneao/sentence_encoders/bert_config.json \
    --save /scratch/gobi2/stephaneao/trained_berts/sd/ \
    --model-type 'sd' \
    --modes 'mlm,sd' \
    --incremental False \
    --track-results True
