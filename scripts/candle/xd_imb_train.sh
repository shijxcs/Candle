#!/bin/bash

# cd ../..

# custom config
DATA=./data
TRAINER=Candle

DATASET=imagenet
DATASET_NEW=$1
CFG=$2
SCALE=$3
MASK=$4
VIR=$5
EPOCHS=$6
SUB=all
IMB=0.01

SHOTS=100

for SEED in 1 2 3
do
  DIR=output/cross-dataset/${DATASET}_imbratio${IMB}/${MASK}_${TRAINER}_${SCALE}/target_${DATASET_NEW}/${CFG}/epoch${EPOCHS}/seed${SEED}
  if [ -d "$DIR" ]; then
      echo "Oops! The results exist at ${DIR} (so skip this job)"
  else
      python train.py \
      --root ${DATA} \
      --seed ${SEED} \
      --trainer ${TRAINER} \
      --dataset-config-file configs/datasets/${DATASET}.yaml \
      --dataset-new-config-file configs/datasets/${DATASET_NEW}_new.yaml \
      --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
      --output-dir ${DIR} \
      DATASET.NUM_SHOTS ${SHOTS} \
      DATASET.SUBSAMPLE_CLASSES ${SUB} \
      DATASET.IMBALANCE_RATIO ${IMB} \
      OPTIM.MAX_EPOCH ${EPOCHS} \
      TRAINER.SCALE ${SCALE} \
      TRAINER.MASK ${MASK} \
      TRAINER.VIR_INIT ${VIR} \
      TRAINER.PHASE train \
      TRAINER.TASK XD
  fi
done
