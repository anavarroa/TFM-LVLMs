#!/bin/bash

### REMEMBER THAT THIS SCRIPT MUST BE RUN WITH LLAVA ENVIRONMENT ACTIVATED ###
### ALSO, jq MUST BE INSTALLED IN THE CONDA ENVIRONMENT WITH conda install -c conda-forge jq ###

# Explicitly setting the locale to en_US.UTF-8 (point instead of comma for decimals)
export LC_ALL=C

# Specify the path you want to loop through
root_path="/datassd/proyectos/tfm-alvaro"

# Check if lora_res already exists
if [ ! -d "$root_path/lora_res" ]; then
    # If the directory does not exist, create it
    mkdir -p "$root_path/lora_res"
    echo "Directory created at: $root_path/lora_res"
else
    echo "Directory already exists at: $root_path/lora_res"
fi

# Accumulation of epochs
acc_epoch=0.1

# Loop through all directories in the specified path
for dir in "$root_path/res"/*; do
    # Check if it is a directory
    if [ -d "$dir" ]; then
        # Extract learning rate
        # The -F'_' option sets the field separator to _, and {print $NF} prints the last field
        values=$(echo "$dir" | awk -F'_' '{print $(NF-2), $(NF-1), $NF}')
        read -r lora_r gradient_accumulation_steps learning_rate <<< "$values"
        for checkpoint in "$dir"/*; do
            if [ -d "$checkpoint" ]; then
                echo "Processing checkpoint: $checkpoint"
                # Take basenames for training directory and checkpoint
                base_dir=$(basename "$dir")
                base_checkpoint=$(basename "$checkpoint")
                echo "$base_dir"
                echo "$base_checkpoint"

                destiny_path="$root_path/lora_res/${base_dir}_${base_checkpoint}"
                # Check if lora_res already exists
                mkdir "$destiny_path"
                echo "Directory created at: $destiny_path"
                
                # Copy into new lora results directory
                cp -r "$checkpoint" "$destiny_path/$base_checkpoint"
                # Extract epoch limit from trainer_state.json
                epoch=$(jq -r '.epoch' "$checkpoint/trainer_state.json")
                rounded_epoch=$(printf "%.2f" "$epoch")
                epoch_limit=$(echo "$rounded_epoch + $acc_epoch" | bc)
                echo "Epoch limit for this checkpoint: $epoch_limit"

                # deepspeed training
                deepspeed "$root_path/model/LLaVA/llava/train/train_mem.py" \
                    --lora_enable True --lora_r $lora_r --lora_alpha 256 --mm_projector_lr 2e-5 \
                    --deepspeed "$root_path/model/LLaVA/scripts/zero3.json" \
                    --model_name_or_path "liuhaotian/llava-v1.5-7b" \
                    --version v1 \
                    --data_path "$root_path/data/sets/data_train.json" \
                    --image_folder "$root_path/data/imagenes" \
                    --vision_tower "openai/clip-vit-large-patch14-336" \
                    --mm_projector_type mlp2x_gelu \
                    --mm_vision_select_layer -2 \
                    --mm_use_im_start_end False \
                    --mm_use_im_patch_token False \
                    --image_aspect_ratio pad \
                    --group_by_modality_length True \
                    --bf16 True \
                    --output_dir $destiny_path \
                    --num_train_epochs $epoch_limit \
                    --per_device_train_batch_size 2 \
                    --per_device_eval_batch_size 4 \
                    --gradient_accumulation_steps $gradient_accumulation_steps \
                    --evaluation_strategy "no" \
                    --save_strategy "steps" \
                    --save_steps 300000 \
                    --save_total_limit 1 \
                    --learning_rate $learning_rate \
                    --weight_decay 0. \
                    --warmup_ratio 0.03 \
                    --lr_scheduler_type "cosine" \
                    --logging_steps 1 \
                    --tf32 True \
                    --model_max_length 2048 \
                    --gradient_checkpointing True \
                    --dataloader_num_workers 4 \
                    --lazy_preprocess True
                    # --report_to wandb

                # When training ends, remove checkpoint from folder
                # so the next one can be trained
                echo "Removing checkpoint: $destiny_path/$base_checkpoint"
                rm -r "$destiny_path/$base_checkpoint"
            fi
        done
    fi
done