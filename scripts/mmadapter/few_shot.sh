#!/bin/bash

# custom config
DATA=PATH_TO_YOUR_DATASETS
TRAINER=MultiModalAdapter
SHOTS=16

DATASET=$1

for SHOTS in 1 2 4 8 16
do
    if [ $DATASET == "imagenet" ]; then
        CFG=vit_b16_ep5_imnet
    else
        CFG=vit_b16_ep5
    fi
    for SEED in 1 2 3
    do
        DIR=output_mmadapter/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Oops! The results exist at ${DIR} (so skip this job)"
        else
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            DATASET.NUM_SHOTS ${SHOTS} \
            TASK FS
        fi
    done
    python3 parse_test_res.py output_mmadapter/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/ 
done