#!/bin/bash

# custom config
DATA="/kaggle/working/PromptKDLOGIT/datasets/data"
TRAINER=PromptKD

DATASET=$1  # 'imagenet' 'caltech101' 'dtd' 'eurosat' 'fgvc_aircraft' 'oxford_flowers' 'food101' 'oxford_pets' 'stanford_cars' 'sun397' 'ucf101'
SEED=$2

CFG=vit_b16_c2_ep20_batch8_4+4ctx
SHOTS=0

DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed_${SEED}

# 根据数据集自动设置KD_WEIGHT
case $DATASET in
    fgvc_aircraft|oxford_flowers|dtd)
        KD_WEIGHT=200.0
        ;;
    *)
        KD_WEIGHT=1000.0  # 其他数据集使用1000.0
        ;;
esac

echo "使用数据集: $DATASET，设置KD_WEIGHT为: $KD_WEIGHT"

CUDA_VISIBLE_DEVICES=0 python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    TRAINER.MODAL base2novel \
    TRAINER.PROMPTKD.TEMPERATURE 1.0 \
    TRAINER.PROMPTKD.KD_WEIGHT ${KD_WEIGHT} \
    TRAINER.PROMPTKD.LOGIT_STANDARDIZATION=True \
    TRAINER.PROMPTKD.ADAPTIVE_TEMPERATURE True
    
