#!/bin/bash

# custom config
DATA=PATH_TO_YOUR_DATASETS
TRAINER=PromptSRC

DATASET=imagenet

CFG=vit_b16_c2_ep20_batch4_4+4ctx_cross_datasets

SHOTS=16

for SEED in 1 2 3
do
    DIR=output_promptsrc/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
python3 parse_test_res.py output_promptsrc/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/