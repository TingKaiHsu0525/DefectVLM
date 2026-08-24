#!/bin/bash

# ChatGLM-6B counterpart of scripts/train_wm811k.sh.
# --vicuna_ckpt_path is reused to point at a local ChatGLM-6B checkpoint directory
# (must contain modeling/tokenizer files + weight shards, e.g. a clone of
# https://huggingface.co/THUDM/chatglm-6b).

deepspeed --include localhost:1 --master_port 29600 train_wm811k_chatglm.py \
    --model openchatglm_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/chatglm_ckpt/chatglm-6b/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --imagebind_lora_ckpt_dir ./ckpt/wm811k_stage1/50_epochs_lr=1e-4_rank=16/\
    --imagebind_lora_rank 16 \
    --imagebind_load_heads \
    --imagebind_load_postprocessors \
    --imagebind_lora_postfix _wm811_last \
    --max_tgt_len 1024\
    --data_path  ../data/pandagpt4_visual_instruction_data.json\
    --image_root_path ../data/images/\
    --save_path  ./ckpt/train_wm811k_chatglm/\
    --log_path ./ckpt/train_wm811k_chatglm/log_rest/\
    --do_eval \
    --eval_every 1 \
    --save_best

# See scripts/train_mixedwm38_chatglm.sh for why --delta_ckpt_path still points at the
# Vicuna-era PandaGPT delta (safe partial warm-start under strict=False loading).
