#!/bin/bash

# ChatGLM-6B counterpart of scripts/train_mixedwm38.sh.
# --vicuna_ckpt_path is reused to point at a local ChatGLM-6B checkpoint directory
# (must contain modeling/tokenizer files + weight shards, e.g. a clone of
# https://huggingface.co/THUDM/chatglm-6b).

deepspeed --include localhost:1 --master_port 28100 train_mixedwm38_chatglm.py \
    --model openchatglm_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --vicuna_ckpt_path ../pretrained_ckpt/chatglm_ckpt/chatglm-6b/\
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt\
    --imagebind_lora_ckpt_dir ./ckpt/mixedwm38_stage1/50_epochs_lr=1e-4_rank=16/\
    --imagebind_lora_rank 16 \
    --imagebind_load_heads \
    --imagebind_load_postprocessors \
    --imagebind_lora_postfix _mixed_last \
    --max_tgt_len 1024\
    --data_path  ../data/pandagpt4_visual_instruction_data.json\
    --image_root_path ../data/images/\
    --save_path  ./ckpt/train_Mixedwm38_chatglm/\
    --log_path ./ckpt/train_Mixedwm38_chatglm/log_rest/\
    --do_eval \
    --eval_every 1 \
    --save_best

# NOTE on --delta_ckpt_path: model/agent.py's DeepSpeedAgent.__init__ unconditionally
# calls load_state_dict(delta_ckpt, strict=False), so *some* valid checkpoint file must
# be passed (agent.py is intentionally left untouched). The existing PandaGPT delta (or
# any prior Vicuna-based ckpt/*/trainable_params.pt) is safe to reuse here: its
# `llama_model.*` keys simply won't match ChatGLM's parameter names and are silently
# skipped, while `llama_proj`/`image_decoder`/`prompt_learner`/`soft_prompt` keep the
# same attribute names *and* shapes (ChatGLM-6B hidden_size=4096, same as Vicuna-7B) in
# model/openchatglm.py, so those DO transfer as a warm start.
