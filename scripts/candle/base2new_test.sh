#!/bin/bash

# cd ../..

# custom config
DATA=./data
TRAINER=Candle

DATASET=$1
CFG=$2
SCALE=$3
MASK=$4
VIR=$5
LOADEP=$6
SUB=$7

SHOTS=16

for SEED in 1 2 3
do
  COMMON_DIR=${CFG}/epoch${LOADEP}/seed${SEED}
  MODEL_DIR=output/base2new/${DATASET}/${MASK}_${TRAINER}_${SCALE}/train_base/${COMMON_DIR}
  DIR=output/base2new/${DATASET}/${MASK}_${TRAINER}_${SCALE}/test_${SUB}/${COMMON_DIR}
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
      --load-epoch ${LOADEP} \
      --eval-only \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES ${SUB} \
      TRAINER.MASK ${MASK} \
      TRAINER.SCALE ${SCALE} \
      TRAINER.VIR_INIT ${VIR} \
      TRAINER.PHASE "test"
  fi
done