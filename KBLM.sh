export CUDA_VISIBLE_DEVICES=0 # ${X_SGE_CUDA_DEVICE}
# export PYTHONPATH="/home/dawna/gs534/fairseq:$PYTHONPATH"
# export PATH="/home/dawna/gs534/fairseq/torchnew/bin:$PATH"
# export PATH="/home/dawna/gs534/Software/anaconda3/bin:$PATH"

mkdir -p LOGs
mkdir -p models

exp_no=1
dataset=SWBD_NE
tag=KBforcing

python train_with_dataloader_debug.py \
    --data ${PWD}/data/${dataset} \
    --cuda \
    --emsize 256 \
    --nhid 1024 \
    --nhid_class 1024 \
    --dropout 0.5 \
    --rnndrop 0.25 \
    --epochs 30 \
    --lr 10.0 \
    --tagloss-scale 0.0 \
    --attloss-scale 0.1 \
    --clip 0.25 \
    --nlayers 1 \
    --batch_size 24 \
    --bptt 36 \
    --pre_epochs 0 \
    --wdecay 1e-6 \
    --model LSTM \
    --reset 0 \
    --logfile LOGs/vanillaLSTM_${dataset}_${exp_no}.classKBlstm_enc_${tag}.log \
    --save models/model.${dataset}_${exp_no}.classKBlstm_enc_${tag}.pt \
    --useKB \
    --rampup \
    --gating \
    --nhop 1 \
    # --from-pretrain \
    # --pretrain-dim 768 \
    # --log-interval 10 \
    # --use_dsc \
    # --gamma 0.1 \
    # --eval_batch_size 1\
    # --stream_out \
