#!/bin/bash

# Extract text from LibriSpeech .trans.txt files only for training sets. 
# 
# This command strips IDs.
# 
# Usage: ./extract_libri_text.sh <data_dir> <output_file>

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <data_dir> <output_file>"
  exit 1
fi

DATA_DIR=$1
OUTPUT_FILE=$2

# Ensure the output file is fresh/empty before appending
> "$OUTPUT_FILE"

# Find files only within the train-* directories and extract text.
find "$DATA_DIR" -maxdepth 4 -path "*/train-*" -name "*.trans.txt" \
    -exec cut -d' ' -f2- {} + >> "$OUTPUT_FILE"
