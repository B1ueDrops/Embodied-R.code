#!/bin/bash

# Set model paths
VISION_MODEL="Qwen/Qwen2.5-Vl-72B-Instruct"  # Vision model path
REASONING_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"   # Reasoning model path

# Set parameters
MAX_TOKENS=4096                # Max output tokens for reasoning module
TEMPERATURE=0.7                # Temperature for reasoning module
VISION_MAX_TOKENS=6144         # Max output tokens for vision module
VISION_TEMPERATURE=0.1         # Temperature for vision module

# Run interactive mode
python infer_our/video_chat_cli.py \
  --interactive \
  --vision_model_path "$VISION_MODEL" \
  --reasoning_model_path "$REASONING_MODEL" \
  --max_tokens $MAX_TOKENS \
  --temperature $TEMPERATURE \
  --vision_max_tokens $VISION_MAX_TOKENS \
  --vision_temperature $VISION_TEMPERATURE

echo "Interactive video chat session ended"
