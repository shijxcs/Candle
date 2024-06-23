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
IMB=0.01

SHOTS=100
SUB=all

for SEED in 1 2 3
do
  COMMON_DIR=${CFG}/epoch${LOADEP}/seed${SEED}
  MODEL_DIR=output/cross-dataset/imagenet_imbratio${IMB}/${MASK}_${TRAINER}_${SCALE}/target_${DATASET}/${COMMON_DIR}
  DIR=output/cross-dataset/${DATASET}_imbratio${IMB}/${MASK}_${TRAINER}_${SCALE}/${COMMON_DIR}
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
      DATASET.IMBALANCE_RATIO ${IMB} \
      TRAINER.MASK ${MASK} \
      TRAINER.SCALE ${SCALE} \
      TRAINER.VIR_INIT ${VIR} \
      TRAINER.PHASE "test" \
      TRAINER.TASK XD
  fi
done