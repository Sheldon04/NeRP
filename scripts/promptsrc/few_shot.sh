#!/bin/bash

# custom config
DATA=PATH_TO_YOUR_DATASETS
TRAINER=PromptSRC
SHOTS=16

DATASET=$1

for SHOTS in 1 2 4 8 16
do
    CFG=vit_b16_c2_ep50_batch4_4+4ctx_few_shot
    fi
    for SEED in 1 2 3
    do
        DIR=output_promptsrc/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
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
    python3 parse_test_res.py output_promptsrc/few_shot/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/ 
done