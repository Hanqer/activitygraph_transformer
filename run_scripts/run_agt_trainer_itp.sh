#!/bin/bash

DATASET=thumos
NCLASSES=20
SR=4
NSTEPS=1
EPOCHS=3000
BATCHSIZE=3
LR=1e-5
POSEMB=learned
LRJOINER=1e-5
SAVEFREQ=300
NQUERIES=300
NENCLAYERS=4
NDECLAYERS=4
HDIM=256
NHEADS=4
NPOSEMB=256
DROPOUT=0
LRDROP=1200
WDECAY=1e-5
CLIPNORM=0
NF=256
EVALFREQ=15

OUT="output/checkpoints_"${FOLDER_SUFFIX}"/checkpoints_numqueries"${NQUERIES}"_lr"$LR"_lrdrop"${LRDROP}"_dropout"${DROPOUT}"_clipmaxnorm"${CLIPNORM}"_weightdecay"${WDECAY}"_posemb"$POSEMB"_lrjoiner"${LRJOINER}"_nheads"${NHEADS}"_nenclayers"${NENCLAYERS}"_ndeclayers"${NDECLAYERS}"_hdim"${HDIM}"_sr"${SR}"_batchsize"${BATCHSIZE}"_nposembdict"${NPOSEMB}"_numinputs"${NF}

LOGDIR="output/logs"
LOG=${LOGDIR}"/log_numqueries"${NQUERIES}"_lr"$LR"_lrdrop"${LRDROP}"_dropout"${DROPOUT}"_clipmaxnorm"${CLIPNORM}"_weightdecay"${WDECAY}"_posemb"$POSEMB"_lrjoiner"${LRJOINER}"_nheads"${NHEADS}"_nenclayers"${NENCLAYERS}"_ndeclayers"${NDECLAYERS}"_hdim"${HDIM}"_sr"${SR}"_batchsize"${BATCHSIZE}"_nposembdict"${NPOSEMB}"_numinputs"${NF}".log"

mkdir -p ${LOGDIR}
exec &> >(tee -a "${LOG}")
echo Logging output to "${LOG}"

DATA=$PWD"/../../data/thumos"

python -m torch.distributed.launch --nproc_per_node 8 src/main.py --dataset ${DATASET} --data_root ${DATA} --model "agt" --features "thumos_i3d_rgb" --batch_size ${BATCHSIZE} --enc_layers ${NENCLAYERS} --dec_layers ${NDECLAYERS} --num_queries ${NQUERIES} --nheads ${NHEADS} --dropout ${DROPOUT} --weight_decay ${WDECAY} --clip_max_norm ${CLIPNORM} --num_inputs ${NF} --num_pos_embed_dict ${NPOSEMB} --hidden_dim ${HDIM} --num_workers 0 --num_classes ${NCLASSES} --step_size ${NSTEPS} --sample_rate ${SR}  --lr ${LR} --lr_drop ${LRDROP} --epochs ${EPOCHS} --output_dir ${OUT} --position_embedding ${POSEMB} --lr_joiner ${LRJOINER} --save_checkpoint_every ${SAVEFREQ} --dim_feedforward $(( 4 * ${HDIM} )) --set_cost_segment 5 --set_cost_siou 3 --segment_loss_coef 5 --siou_loss_coef 3 --cuda --eval_interval ${EVALFREQ}


