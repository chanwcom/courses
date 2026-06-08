# LibriSpeech SentencePiece Tokenizer Training Pipeline

This project provides an automated pipeline to extract text from the LibriSpeech dataset and train multiple **SentencePiece** tokenizers (Unigram and BPE) with various vocabulary sizes.

---

## Setup & Execution Guide

### Step 1: Download the Full LibriSpeech Dataset and the Normalized LibriSpeech Text Corpus

 - LibriSpeech ASR dataset can be found at:
```
https://www.openslr.org/11
```
 - LibriSpeech LM Corpus dataset can be found at:
```
https://www.openslr.org/12
```

 - First, download and extract the LibriSpeech ASR (audio/text) data (e.g., `train-clean-100`) to your local machine.

```bash
# Create a data directory
mkdir -p ./data

# Download LibriSpeech train-clean-100 (approx. 6.3 GB)
wget [http://www.openslr.org/resources/12/train-clean-100.tar.gz](http://www.openslr.org/resources/12/train-clean-100.tar.gz)

# Extract the dataset
tar -xzvf train-clean-100.tar.gz -C ./data
```

LibriSpeech transcripts are stored in .trans.txt files containing both utterance IDs and text. Use the extraction script to strip the IDs and merge all text into a single file for training.

Make sure the output path matches the configuration in your scripts:

```
# Ensure the script is executable
chmod +x extract_libri_text.sh

# Run the extraction script
./extract_libri_text.sh ./data/LibriSpeech ./src/data/tokenizers/resources/libri_raw.txt
```

 - Download the Normalized Text Corpus
To train a robust tokenizer, we will use the external normalized text corpus provided by OpenSLR (LibriSpeech language model training text). This file contains a large amount of clean text data suitable for vocabulary building.

URL: OpenSLR 11 (LibriSpeech LM corpus)

File to download: librispeech-lm-norm.txt.gz

Download and extract the text corpus:

``````
mkdir -p data/text_corpus
cd data/text_corpus

wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
gunzip librispeech-lm-norm.txt.gz
``````

- Concatenate `libri_raw.txt` and `librispeech-lm-norm.txt.gz` to make the final text corpus.

### Step 2: Train the Tokenizers
You can train models using either Unigram or BPE algorithms. The training script will automatically iterate through the predefined vocabulary sizes: 32, 128, 512, 2048, and 8192.

Before running, open run.sh and ensure the OUTPUT_DIR path exists or is set to your preferred directory.

Option A: Train Unigram Models (Default)
Ensure TOKENIZATION_TYPE=unigram is set inside run.sh, then run:
```
chmod +x run.sh
./run.sh
```

Option B: Train BPE Models
Open run.sh, change the tokenization type variable to bpe:
```
TOKENIZATION_TYPE=bpe
```
Then execute the script:

```
./run.sh
```

### Step 3: Verify Output Artifacts
Once run.sh finishes running successfully, verify that the tokenizer models and vocabularies have been properly generated in your specified OUTPUT_DIR.

Navigate to your output directory and list the files:

```
ls -l /mnt/kioxia_exeria/home/chanwcom/tmp/models/asr/
```

You should see a pair of .model and .vocab files for each vocabulary size:

- librispeech_unigram_32.model & librispeech_unigram_32.vocab
- librispeech_unigram_128.model & librispeech_unigram_128.vocab
- librispeech_unigram_512.model & librispeech_unigram_512.vocab
- librispeech_unigram_2048.model & librispeech_unigram_2048.vocab
- librispeech_unigram_8192.model & librispeech_unigram_8192.vocab

### Step 4: Test and Load Tokenizer in Python
To verify that your newly trained models are working correctly within your code, load them using the sentencepiece library and tokenize a sample sentence.

Create a simple script or open an interactive Python shell:
```
import sentencepiece as spm

# Path to one of the trained models
model_path = "/mnt/kioxia_exeria/home/chanwcom/tmp/models/asr/librispeech_unigram_2048.model"

# Load the model
sp = spm.SentencePieceProcessor(model_file=model_path)

# Test tokenization
text = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
print("Tokens:", sp.encode_as_pieces(text))
print("Token IDs:", sp.encode_as_ids(text))
```

📂 Directory Structure
The pipeline consists of the following components:

- extract_libri_text.sh: A shell script that recursively searches for LibriSpeech transcription files (*.trans.txt), removes the utterance IDs from each line, and aggregates the raw text.

- train_spm.py: A Python script that trains a SentencePiece tokenizer using specified parameters such as model type, vocabulary size, character coverage (1.0), and special control tokens (pad, eos, bos, unk).

- run.sh: An automation script that iterates through multiple vocabulary sizes (32, 128, 512, 2048, 8192) and triggers the python training script sequentially.
