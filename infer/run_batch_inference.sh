#!/bin/bash

# Set default parameters
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
INPUT_FILE="results/inter/test_data.json"
OUTPUT_FILE="results/infer/inference_result_$(date +%Y%m%d_%H%M%S).json"
BATCH_SIZE=1
MAX_TOKENS=3096
ADAPTER_PATH=""

# Function to display help information
show_help() {
    echo "Batch Video Q&A Inference Script"
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -m, --model MODEL_PATH      Specify model path (default: $MODEL)"
    echo "  -i, --input INPUT_FILE      Specify input JSON file (default: $INPUT_FILE)"
    echo "  -o, --output OUTPUT_FILE    Specify output JSON file (default: timestamp-based output file)"
    echo "  -b, --batch BATCH_SIZE      Specify batch size (default: $BATCH_SIZE)"
    echo "  -t, --tokens MAX_TOKENS     Specify maximum number of tokens to generate (default: $MAX_TOKENS)"
    echo "  -a, --adapter ADAPTER_PATH  Specify LoRA adapter path (optional)"
    echo "  -h, --help                  Display this help information"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -t|--tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        -a|--adapter)
            ADAPTER_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Ensure output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python infer/batch_inference.py --model \"$MODEL\" --input_file \"$INPUT_FILE\" --output_file \"$OUTPUT_FILE\" --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS"

# Add adapter path to command if provided
if [ -n "$ADAPTER_PATH" ]; then
    CMD="$CMD --adapter_path \"$ADAPTER_PATH\""
fi

# Display the command to be executed
echo "Executing command: $CMD"
echo "Start time: $(date)"

# Execute command
eval $CMD

# Display completion information
echo "Completion time: $(date)"
echo "Results saved to: $OUTPUT_FILE"
