#!/bin/bash
# from here https://huggingface.co/docs/trl/en/cpo_trainer
# unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit

# /lus/eagle/projects/FoundEpidem/bhsu/huggingface/.cache/accelerate/default_config.yaml 
# ACCELERATE_CONFIG_PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/ml_program/config_files/accelerate_config_stage3.yaml"
ACCELERATE_CONFIG_PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/ml_program/config_files/accelerage_config_stage2.yaml"
# ACCELERATE_CONFIG_PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/ml_program/config_files/ddp_accelerate.yaml"
FILE_PATH="/lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/ml_program/train.py"

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file $ACCELERATE_CONFIG_PATH --num_machines=2 --num_processes=4 $FILE_PATH \
--model_name_or_path unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit \
--dataset_name_or_path coseal/CodeUltraFeedback_binarized \
--checkpoint_dir /lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/checkpoints \
--train_config_path /lus/eagle/projects/FoundEpidem/bhsu/2024_research/ml-programming-winter-2025/ml_program/config_files/polaris_train_config.yaml
