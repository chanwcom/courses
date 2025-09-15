#!/bin/bash

# Command line arguments
SRC_DIR="$1"
DST_DIR="$2"
TEXT_FILE="$3"

# Check if all arguments are provided
if [ -z "$SRC_DIR" ] || [ -z "$DST_DIR" ] || [ -z "$TEXT_FILE" ]; then
    echo "Usage: $0 <SRC_DIR> <DST_DIR> <TEXT_FILE>"
    exit 1
fi

# Create destination directory if it doesn't exist
mkdir -p "$DST_DIR"

# Check if sox is installed
if ! command -v sox &> /dev/null; then
    echo "sox is not installed or not in PATH."
    exit 1
fi

# Read the text file and extract the first column (wav filenames)
awk '{print $1}' "$TEXT_FILE" | while read wav_file; do
    # Check if the wav file exists in the source directory
    if [ -f "$SRC_DIR/$wav_file" ]; then
        base=$(basename "$wav_file" .wav)
        echo "Converting $SRC_DIR/$wav_file to $DST_DIR/$base.flac ..."
        sox "$SRC_DIR/$wav_file" "$DST_DIR/$base.flac"
    else
        echo "$SRC_DIR/$wav_file does not exist."
    fi
done

echo "Conversion done."

