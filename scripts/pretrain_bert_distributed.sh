#!/bin/bash

WORLD_SIZE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS olfmlm/pretrain_bert.py "$@"



#    --batch-size 32 \
#    --tokenizer-type BertWordPieceTokenizer \
#    --cache-dir cache_dir \
#    --tokenizer-model-type bert-base-uncased \
#    --vocab-size 30522 \
#    --train-data wikipedia \
#    --presplit-sentences \
#    --loose-json \
#    --text-key text \
#    --split 1000,1,1 \
#    --lazy-loader \
#    --max-preds-per-seq 80 \
#    --seq-length 128 \
#    --max-position-embeddings 512 \
#    --train-iters 100000 \
#    --lr 0.0001 \
#    --lr-decay-style linear \
#    --warmup .01 \
#    --weight-decay 1e-2 \
#    --clip-grad 1.0 \
#    --num-workers 1 \
#    --model-type 'bertmlm' \
#    --track-results True
#    --lr-decay-iters 990000 \
