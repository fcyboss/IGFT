# RM
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage rm \
    --model_name_or_path llama_path \
    --do_train \
    --dataset datasetname \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir llama_sft_checkpoint \
    --output_dir llama_rm1th_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-8 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --fp16



# PPO
CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage ppo \
    --model_name_or_path llama_path \
    --do_train \
    --dataset datasetname \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --resume_lora_training False \
    --checkpoint_dir llama_sft_checkpoint \
    --reward_model llama_rm1th_checkpoint \
    --output_dir llama_ppo1th_checkpoint \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-7 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --bf16
