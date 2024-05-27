#!/bin/bash

# Set default values for input and output directories
input_dir="../0_outputs/hex_category_1/stage_4_batch_request/"
output_dir="../0_outputs/hex_category_1/stage_5_plotting"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            input_dir="$2"
            shift 2
            ;;
        --output_dir)
            output_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Run the Python script with the specified arguments
python stage_8_plotting_gpt_eval.py --input_dir "$input_dir" --output_dir "$output_dir"

# # For keyword eval in the future
# python script.py --input_dir "$INPUT_DIR" --output_dir "$OUTPUT_DIR"