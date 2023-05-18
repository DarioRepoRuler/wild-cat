#!/usr/bin/env bash

set -x

EXP_DIR=exps/OWOD_t1_new_split
PY_ARGS=${@:1}

CUDA_VISIBLE_DEVICES="0" python -u main_open_world.py \
    --output_dir ${EXP_DIR} --dataset owod --num_queries 100 --eval_every 1 \
    --PREV_INTRODUCED_CLS 0 --CUR_INTRODUCED_CLS 19 --train_set 't1_train' --test_set 'test' --num_classes 81 \
    --unmatched_boxes --epochs 45 --top_unk 5 --featdim 1024 --NC_branch --nc_loss_coef 0.1 --nc_epoch 9 \
    --enable_adaptive_pseudo --pseudo_store_path 'exps/loss_memory'  --adaptive_update_Iter 300 --memory_length 100 \
    --backbone 'dino_resnet50' \
    ${PY_ARGS}

EXP_DIR=exps/OWOD_t2_new_split
PY_ARGS=${@:1}