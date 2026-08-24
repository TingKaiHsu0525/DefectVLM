#!/bin/bash 

deepspeed --include localhost:1 --master_port 29500 train_wm811k.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --imagebind_lora_ckpt_dir ./ckpt/wm811k_stage1_rp/\
    --imagebind_lora_rank 16 \
    --imagebind_load_heads \
    --imagebind_load_postprocessors \
    --imagebind_lora_postfix _last \
    --max_tgt_len 1024\
    --data_path  ../data/pandagpt4_visual_instruction_data.json\
    --image_root_path ../data/images/\
    --save_path  ./ckpt/train_wm811k\
    --log_path ./ckpt/train_wm811k/log_rest/\
    --do_eval \
    --eval_every 1 \
    --save_best


# original
# --lora_ckpt_path ./ckpt/imagebind_lora_encoder_wm811k.pt \

# use imagebind lora
    # --imagebind_lora_ckpt_dir ./ckpt/wm811k_stage1/50_epochs_lr=1e-4_rank=16 \
    # --imagebind_lora_rank 16 \
    # --imagebind_load_heads \
    # --imagebind_load_postprocessors \
