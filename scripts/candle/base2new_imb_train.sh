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
IMB=$6
EPOCHS=$7
SHOTS=$8

for SEED in 1 2 3
do
  DIR=output/base2new-imb/${DATASET}_imbratio${IMB}/${MASK}_${TRAINER}_${SCALE}/train_base/${CFG}/epoch${EPOCHS}/seed${SEED}
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
      DATASET.SUBSAMPLE_CLASSES base \
      DATASET.IMBALANCE_RATIO ${IMB} \
      OPTIM.MAX_EPOCH ${EPOCHS} \
      TRAINER.SCALE ${SCALE} \
      TRAINER.MASK ${MASK} \
      TRAINER.PHASE train
  fi
done
