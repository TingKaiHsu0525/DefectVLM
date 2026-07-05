CUDA_VISIBLE_DEVICES=1 python train.py \
    --batch_size 16 \
    --max_epochs 50 \
    --num_workers 4 \
    --self_contrast \
    --lora \
    --lora_rank 16 \
    --lora_modality_names vision text \
    --lr 1e-4 \
    --datasets mixedwm38 \
    --mixedwm38_train_json .datasets/train.json \
    --mixedwm38_val_json .datasets/val.json \
    --mixedwm38_text_mode label \
    --device cuda:0 \
    --headless \
    --loggers tensorboard 


# mixed config
    # --datasets mixedwm38 \
    # --mixedwm38_train_json .datasets/train.json \
    # --mixedwm38_val_json .datasets/val.json \
    # --mixedwm38_text_mode label \

# wm811k config
    # --datasets wm811k \
    # --wm811k_train_json .datasets/WM-811k/train.json \
    # --wm811k_val_json .datasets/WM-811k/val.json \
    # --wm811k_text_mode label \