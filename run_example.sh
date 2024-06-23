# base-to-new generalization

DATASET=caltech101
CFG=vit_b16_bs128_lr0.0003
SCALE=1.0 # controls the scale between visual and text temperatures
MASK=none # for ablation on attn
VIR=text # virtual prototype init
IMB=0.02 # imbalance ratio, set to 1.0 for balanced scenarios; fixed at 0.01 for imagenet
EPOCHS=50
SHOTS=100 # maximum number of samples per class

# training
bash scripts/candle/base2new_imb_train.sh ${DATASET} ${CFG} ${SCALE} ${MASK} ${VIR} ${IMB} ${EPOCHS} ${SHOTS}
# testing
bash scripts/candle/base2new_imb_test.sh ${DATASET} ${CFG} ${SCALE} ${MASK} ${VIR} ${IMB} ${EPOCHS} ${SHOTS} base
bash scripts/candle/base2new_imb_test.sh ${DATASET} ${CFG} ${SCALE} ${MASK} ${VIR} ${IMB} ${EPOCHS} ${SHOTS} new
