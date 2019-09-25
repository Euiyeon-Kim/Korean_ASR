#!/bin/sh

export CUDA_VISIBLE_DEVICES="0"

BATCH_SIZE=16
WORKER_SIZE=4
MAX_EPOCHS=10

python3 ./train.py --lr 1e-3 --batch_size $BATCH_SIZE --workers\
 $WORKER_SIZE --use_attention --bidirectional --max_epochs $MAX_EPOCHS --hidden_size 512 --layer_size 4
