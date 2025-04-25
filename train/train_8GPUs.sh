#!/bin/bash
SYSTEM_PROMPT="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> \n<answer> answer here </answer>. Ensure that your answer is consistent with and directly derived from your thinking process, maintaining logical coherence between the two sections. User: . Assistant:"
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model "Qwen/Qwen2.5-VL-3B-Instruct" \
    --train_type full \
    --torch_dtype bfloat16 \
    --system "${SYSTEM_PROMPT}" \
    --dataset "results/inter/train_data_grpo.jsonl" \
    --val_dataset "results/inter/val_data_grpo.jsonl" \
    --dataloader_num_workers 8 \
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
    --external_plugins train/consistency_reward_API.py train/choice_accuracy_reward.py \
    --beta 0.001 \
    --temperature 1.0 \
    --num_generations 8 \
    --num_iterations 1 \
    --use_vllm true \
    --vllm_gpu_memory_utilization 0.4 \
    --tensor_parallel_size 4 \
    --deepspeed zero3 \
    --num_infer_workers 8 \
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
