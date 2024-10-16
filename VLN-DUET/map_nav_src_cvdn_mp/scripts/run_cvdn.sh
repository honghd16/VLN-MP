DATA_ROOT=../datasets

train_alg=dagger

features=vitbase
ft_dim=768
obj_features=vitbase
obj_ft_dim=768

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed}
name=${name}-init.aug.45k

outdir=${DATA_ROOT}/CVDN/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset cvdn
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert
      --use_player_path

      --enc_full_graph
      --graph_sprels
      --fusion dynamic

      --expert_policy spl
      --train_alg ${train_alg}

      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2

      --max_action_len 30
      --max_instr_len 200

      --batch_size 8
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2

      --feat_dropout 0.4
      --dropout 0.5

      --gamma 0."

# train
CUDA_VISIBLE_DEVICES='0' python cvdn/main.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file ../datasets/R2R/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap-init.lxmert-aug.speaker/ckpts/model.pt \
      --eval_first \
