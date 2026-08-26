# DefectVLM: A Vicuna-Based Vision Language Model for Wafer Map Defect Understanding

## Installation
```
conda env create -f environment.yml
conda activate DefectVLM
```
Then pip uninstall torch, torchvision, torchaudio and install the veriosn satisfy your CUDA envirnment.
In my envirnment this will work:
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117
```

## Stage1

- Run stage1: 
    ```
    cd Stage1
    bash train.sh
    ```
    You will get the model weight in `.checkpoints`.

## Stage2

- Prepare the pretrained model weights in `pretrained_ckpt` (`imagebind_ckpt`, `vicuna_ckpt`, `pandagpt_ckpt`) following the instructions from AnomalyGPT: <https://github.com/CASIA-LMC-Lab/AnomalyGPT>.
- Before training, move the Stage1 checkpoint to `Stage2/code/ckpt/{dataset}_stage1/` and set `--imagebind_lora_ckpt_dir` in the Stage2 train script accordingly.


- Run stage2: 
    - Train
    ```
    cd code
    # mixedwm38
    bash scripts/train_mixedwm38.sh
    
    # wm811k
    bash scripts/train_wm811k.sh
    ```
    - Inference
    ```
    # mixedwm38
    bash scripts/eval_mixedwm38.sh

    # wm811k
    bash scripts/eval_wm811k.sh
    ```
    Each script runs inference to produce the eval-result JSON, then immediately runs the `cal_acc*.py` on that same file.

    - Inference only
    ```
    # mixedwm38
    python test_mixedwm38.py --output mixedwm38_eval_result.json

    # wm811k 
    python test_wm811k.py --output eval_result_WM-811k.json
    ```
    Inference code will generates json file which containes test dataset answer responses.

    - Calculate accuracy only
    ```
    # mixedwm38
    python cal_acc.py --input mixedwm38_eval_result.json

    # wm811k
    python cal_acc_wm811k.py --input eval_result_WM-811k.json
    ```
    This code will calculate accuracy of the model responses recorded in the given inference
    output json file.