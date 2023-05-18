#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWOD_t3_ft_old_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 40 --CUR_INTRODUCED_CLS 20 --train_set 't3_ft' --test_set 'test' --num_classes 81 \
    --epochs 2 --top_unk 5 --featdim 1024 --nc_loss_coef 0.1 --nc_epoch 9 \
    --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100 \
    --backbone 'dino_resnet50' \
    --pretrain 'exps/OWOD_t3_ft_new_split/new_t3.pth' \
    --eval --viz\
    ${PY_ARGS}

    