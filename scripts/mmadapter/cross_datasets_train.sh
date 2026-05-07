#!/bin/bash

# custom config
DATA=PATH_TO_YOUR_DATASETS
TRAINER=MultiModalAdapter

DATASET=imagenet

CFG=vit_b16_ep1_cross_datasets

SHOTS=16

for SEED in 1 2 3
do
    DIR=output_mmadapter/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
        TASK CD
    fi
done
python3 parse_test_res.py output_mmadapter/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/