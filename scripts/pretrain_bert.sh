#!/bin/bash

RANK=0
WORLD_SIZE=1

python3 sentence_encoders/pretrain_bert.py "$@"


#    --batch-size 32 \
#    --tokenizer-type BertWordPieceTokenizer \
#    --cache-dir cache_dir \
#    --tokenizer-model-type bert-base-uncased \
#    --vocab-size 30522 \
#    --train-data 'wikipedia' \
#    --presplit-sentences \
#    --text-key text \
#    --split 1000,1,1 \
#    --lazy-loader \
#    --max-preds-per-seq 80 \
#    --seq-length 128 \
#    --train-tokens 500000000 \
#    --lr 0.0001 \
#    --lr-decay-style linear \
#    --warmup .01 \
#    --weight-decay 1e-2 \
#    --clip-grad 1.0 \
#    --num-workers 2 \
#    --epochs 2 \
#    --bert-config-file /h/stephaneao/sentence_encoders/bert_config.json \
#    --save /scratch/gobi2/stephaneao/trained_berts/bert/ \
#    --model-type 'bert' \
#    --modes 'mlm,nsp' \
#    --incremental False \
#    --track-results True
