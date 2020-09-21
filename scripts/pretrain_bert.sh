#!/bin/bash

RANK=0
WORLD_SIZE=1

python3 -m sentence_encoders.pretrain_bert "$@"
