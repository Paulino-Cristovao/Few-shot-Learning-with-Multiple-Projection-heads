#
LR=0.001
NUMBER_SAMPLES=1


LATENT_LIST="16 32"

for LATENT in $LATENT_LIST ; do

python pretrain.py --latent $LATENT --lr $LR
python imprint.py --model $LATENT/pretrain_checkpoint/model_best.pth.tar --latent $LATENT --num-sample $NUMBER_SAMPLES


done;
