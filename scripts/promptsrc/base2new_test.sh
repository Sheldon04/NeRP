#!/bin/bash

# custom config
DATA=PATH_TO_YOUR_DATASETS
TRAINER=PromptSRC

DATASET=$1

CFG=vit_b16_c2_ep20_batch4_4+4ctx

SHOTS=16
SUB=new
LOADEP=20

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output_promptsrc/base2new/train_base/${COMMON_DIR}
    DIR=output_promptsrc/base2new/test_${SUB}/${COMMON_DIR}
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
        --model-dir ${MODEL_DIR} \
        --eval-only \
        --load-epoch ${LOADEP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB} \
        TASK B2N
    fi
done
python3 parse_test_res.py output_promptsrc/base2new/test_${SUB}/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/