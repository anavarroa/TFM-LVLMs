import os

# RUTAS
DEEPSPEED_SCRIPT = "deepspeed model/LLaVA/llava/train/train_mem.py"
DEEPSPEED_JSON = "model/LLaVA/scripts/zero3.json"
MODEL_NAME = "liuhaotian/llava-v1.5-7b"
DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data","dataset.json"))
TRAIN_DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data","sets","data_train.json"))
TEST_DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data","sets","data_test.json"))
#VAL_DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data","sets","data_val.json"))
IMAGE_FOLDER = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","data","imagenes"))
VISION_TOWER = "openai/clip-vit-large-patch14-336"
OUTPUT_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","..","..","res"))

# PARÁMETROS:

# > lora_r [8,64]
# > num_train_epochs [1,10]
# > gradient_accumulation_steps [1-10]
# > learning_rate [2e-5,4e-3]

finetune_script = f'''
{DEEPSPEED_SCRIPT} \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed {DEEPSPEED_JSON} \
    --model_name_or_path {MODEL_NAME} \
    --version v1 \
    --data_path {TRAIN_DATA_PATH} \
    --image_folder {IMAGE_FOLDER} \
    --vision_tower {VISION_TOWER} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir {OUTPUT_DIR} \
    --num_train_epochs 10.1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
'''

# conjunto de validación requiero modificar train.py
# report_to wandb para seguimiento en Weights & Biases
# sin LORA falta memoria en GPU


# Borrar cache de CUDA
import torch
torch.cuda.empty_cache()

# Ejecución del stream
import subprocess
print(finetune_script)

result = subprocess.run([finetune_script], shell=True, capture_output=True, text=True)
print(result.stdout)