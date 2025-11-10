import os
import torch
import logging
import json
from datasets import load_dataset
# Using PreTrainedTokenizerFast to load custom files
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerFast
)

# Set logging level to INFO
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# --- 1. CONFIGURATION CONSTANTS (UPPER_SNAKE_CASE) ---
_TRAIN_FILE = "librispeech-lm-norm.txt"
_MODEL_NAME = "librispeech_lm_tinyllama_custom"

# Custom vocabulary and its size (32 tokens)
_CUSTOM_VOCAB = {
    '<pad>': 0, '<s>': 1, '</s>': 2, '<unk>': 3, '|': 4, 'E': 5,
    'T': 6, 'A': 7, 'O': 8, 'N': 9, 'I': 10, 'H': 11, 'S': 12,
    'R': 13, 'D': 14, 'L': 15, 'U': 16, 'M': 17, 'W': 18, 'C': 19,
    'F': 20, 'G': 21, 'Y': 22, 'P': 23, 'B': 24, 'V': 25, 'K': 26,
    "'": 27, 'X': 28, 'J': 29, 'Q': 30, 'Z': 31
}
_VOCAB_SIZE = len(_CUSTOM_VOCAB) # 32

_MAX_SEQUENCE_LENGTH = 128
_NUM_EPOCHS = 3
_PER_DEVICE_BATCH_SIZE = 8
_GRADIENT_ACCUMULATION_STEPS = 2

# --- 2. FILE AND ENVIRONMENT CHECK ---
if not os.path.exists(_TRAIN_FILE):
    LOGGER.error(f"Error: Training file '{_TRAIN_FILE}' not found.")
    LOGGER.error("Ensure the file is downloaded in the current directory.")
    # In a real script, you'd exit here: exit()

# --- 3. CUSTOM TOKENIZER CREATION (REPLACES TRAINING) ---
LOGGER.info("Step 1: Creating custom tokenizer files...")
os.makedirs(_MODEL_NAME, exist_ok=True)

# 1. Save the vocab dictionary as vocab.json
with open(os.path.join(_MODEL_NAME, "vocab.json"), "w") as f:
    json.dump(_CUSTOM_VOCAB, f)

# 2. Create an empty merges.txt (required for BPE-like tokenizers)
with open(os.path.join(_MODEL_NAME, "merges.txt"), "w") as f:
    pass # File is empty

LOGGER.info(
    f"Custom tokenizer files created with size {_VOCAB_SIZE} in "
    f"'{_MODEL_NAME}'."
)

# --- 4. DATA LOADING AND PREPROCESSING ---

# Load the text dataset
LOGGER.info("Step 2: Loading and tokenizing dataset...")
dataset = load_dataset("text", data_files={"train": _TRAIN_FILE})

# Define the custom tokenization function to split by character
def _simple_char_tokenize(self, text):
    """Splits text into characters, mapping to IDs."""
    # Convert to uppercase for case-insensitive matching (common for ASR LMs)
    return [
        t if t in self.vocab else self.unk_token
        for t in text.upper()
    ]

# Load the custom tokenizer using vocab/merges files
tokenizer = PreTrainedTokenizerFast(
    vocab_file=os.path.join(_MODEL_NAME, "vocab.json"),
    merges_file=os.path.join(_MODEL_NAME, "merges.txt"),
    # Set special tokens based on the custom vocab keys
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>",
    # Ensure added tokens are registered
    added_tokens=[
        t for t in _CUSTOM_VOCAB.keys()
        if t not in ["<pad>", "<s>", "</s>", "<unk>"]
    ]
)

# Patch the internal tokenization method for character-level splitting
tokenizer._tokenize = _simple_char_tokenize.__get__(tokenizer)
tokenizer.model_max_length = _MAX_SEQUENCE_LENGTH

def tokenize_function(examples):
    """Tokenizes input text, ensuring truncation."""
    return tokenizer(examples["text"], truncation=True)

# Apply tokenization
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=os.cpu_count() or 1,
    remove_columns=["text"],
)

def group_texts(examples):
    """Groups tokenized texts into contiguous blocks of max length."""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(
        concatenated_examples[list(examples.keys())[0]]
    )

    total_length = (total_length // _MAX_SEQUENCE_LENGTH) * \
                   _MAX_SEQUENCE_LENGTH

    result = {
        k: [
            t[i : i + _MAX_SEQUENCE_LENGTH]
            for i in range(0, total_length, _MAX_SEQUENCE_LENGTH)
        ]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Apply the chunking/grouping function
lm_dataset = tokenized_dataset.map(
    group_texts,
    batched=True,
    num_proc=os.cpu_count() or 1,
)
train_dataset = lm_dataset["train"]
LOGGER.info(
    f"Dataset prep complete. Total training sequences: {len(train_dataset)}"
)

# --- 5. MODEL CONFIGURATION AND INITIALIZATION ---
LOGGER.info("Step 3: Initializing Llama model architecture...")

# Use the custom vocab size and IDs
custom_config = LlamaConfig(
    vocab_size=_VOCAB_SIZE, # 32
    hidden_size=512,
    intermediate_size=512 * 4,
    num_hidden_layers=6,
    num_attention_heads=8,
    max_position_embeddings=_MAX_SEQUENCE_LENGTH,
    rms_norm_eps=1e-6,
    pad_token_id=_CUSTOM_VOCAB['<pad>'],
    bos_token_id=_CUSTOM_VOCAB['<s>'],
    eos_token_id=_CUSTOM_VOCAB['</s>'],
)

model = LlamaForCausalLM(config=custom_config)
LOGGER.info(
    f"Model initialized. Total parameters: {model.num_parameters()}"
)

# --- 6. TRAINING ARGUMENTS AND TRAINER SETUP ---
training_args = TrainingArguments(
    output_dir=f"./{_MODEL_NAME}_results",
    overwrite_output_dir=True,
    num_train_epochs=_NUM_EPOCHS,
    per_device_train_batch_size=_PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=_GRADIENT_ACCUMULATION_STEPS,
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# --- 7. START TRAINING ---
LOGGER.info("Step 4: Starting model training...")
trainer.train()

# --- 8. SAVE FINAL MODEL AND TOKENIZER ---
trainer.save_model(f"./{_MODEL_NAME}_final")
tokenizer.save_pretrained(f"./{_MODEL_NAME}_final")
LOGGER.info("Training complete. Final model and tokenizer saved.")
