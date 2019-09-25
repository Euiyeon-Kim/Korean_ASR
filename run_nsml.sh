#!/bin/sh

BATCH_SIZE=16
WORKER_SIZE=4
GPU_SIZE=1
CPU_SIZE=2
DATASET="sr-hack-2019-dataset"
MAX_EPOCHS=500

nsml run -e train.py -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a \
"--lr 1e-3 --batch_size $BATCH_SIZE --workers $WORKER_SIZE \
--use_attention --bidirectional --max_epochs $MAX_EPOCHS --hidden_size 512 --layer_size 4"
