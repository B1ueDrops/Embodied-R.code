#!/bin/bash
# Configuration Description
# This script is designed for 5-card A800 80GB memory configuration
# GPU Allocation:
# - GPU 0-3: For GRPO training (4-card parallel)
# - GPU 4: For consistency verification model service

# Start Swift-based consistency verification model service
start_consistency_service() {
    echo "Starting Swift-based consistency verification model service..."
    # Give execution permission to the script
    chmod +x train/reward/start_consistency_service.sh 
    # Start the service
    bash train/reward/start_consistency_service.sh 
    # Wait for the service to fully start
    sleep 15
}

# Main function
main() {
    # Start Swift-based consistency verification model service on GPU 4
    start_consistency_service
    
    # Define system prompt
    SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \n<answer> answer here </answer>. Ensure that your answer is consistent with and directly derived from your thinking process, maintaining logical coherence between the two sections. User: . Assistant:"

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    NPROC_PER_NODE=4 \
    swift rlhf \
    --rlhf_type grpo \
    --model "Qwen/Qwen2.5-VL-3B-Instruct" \
    --train_type full \
    --torch_dtype bfloat16 \
    --system "${SYSTEM_PROMPT}" \
    --dataset "results/inter/train_data_grpo.jsonl" \
    --val_dataset "results/inter/val_data_grpo.jsonl" \
    --dataloader_num_workers 4 \
    --num_train_epochs 2 \
    --learning_rate 5e-7 \
    --warmup_ratio 0.05 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --target_modules all-linear \
    --reward_weights 0.7 0.1 0.2 \
    --reward_funcs choice_accuracy format consistency \
    --external_plugins train/reward/consistency_reward_local.py train/reward/choice_accuracy_reward.py \
    --beta 0.001 \
    --temperature 1.0 \
    --num_generations 4 \
    --num_iterations 1 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.6 \
    --tensor_parallel_size 4 \
    --deepspeed zero3 \
    --num_infer_workers 4 \
    --max_length 6144 \
    --max_completion_length 2048 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_only_model true \
    --save_total_limit 200 \
    --output_dir "results/train" \
    --logging_steps 1 \
    --report_to wandb \
    --log_completions true \
    --log_level debug \
    --log_level_replica warning \
    2>&1 | tee "results/train.log"
}
main
