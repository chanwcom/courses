import os
import torch
import logging
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
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
logger = logging.getLogger(__name__)

# --- 1. CONFIGURATION CONSTANTS ---
# Define constants for file paths and hyperparameters according to Google Style (UPPER_SNAKE_CASE)
_TRAIN_FILE = "librispeech-lm-norm.txt"  
_MODEL_NAME = "librispeech_lm_tinyllama_custom"
_VOCAB_SIZE = 30000       # Vocabulary size for the BPE tokenizer
_MAX_SEQUENCE_LENGTH = 128 # Maximum length of input sequences
_NUM_EPOCHS = 3
_PER_DEVICE_BATCH_SIZE = 8 # Batch size per GPU/CPU
_GRADIENT_ACCUMULATION_STEPS = 2 # Steps to accumulate gradients over

# --- 2. FILE AND ENVIRONMENT CHECK ---
if not os.path.exists(_TRAIN_FILE):
    logger.error(f"Error: Training file '{_TRAIN_FILE}' not found.")
    logger.error("Please ensure the file is downloaded and unzipped in the current directory.")
    exit()

# --- 3. TOKENIZER TRAINING AND SAVING ---
# We train a new Byte-level BPE tokenizer on the target corpus.
logger.info("Step 1: Starting tokenizer training...")
custom_tokenizer = ByteLevelBPETokenizer()

# Train the tokenizer, defining special tokens consistent with the Llama architecture.
custom_tokenizer.train(
    files=[_TRAIN_FILE],
    vocab_size=_VOCAB_SIZE,
    min_frequency=2,
    # Standard special tokens for Llama-like models
    special_tokens=["<s>", "</s>", "<unk>", "<pad>"], 
)

# Save the tokenizer model to a dedicated directory
os.makedirs(_MODEL_NAME, exist_ok=True)
custom_tokenizer.save_model(_MODEL_NAME)
logger.info(f"Tokenizer trained and saved to '{_MODEL_NAME}' directory.")


# --- 4. DATA LOADING AND PREPROCESSING ---

# Load the text dataset using the Hugging Face 'text' loader
logger.info("Step 2: Loading and tokenizing dataset...")
dataset = load_dataset("text", data_files={"train": _TRAIN_FILE})

# Load the saved tokenizer into the Hugging Face PreTrainedTokenizerFast format
tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=os.path.join(_MODEL_NAME, "tokenizer.json"),
    unk_token="<unk>",
    bos_token="<s>",
    eos_token="</s>",
    pad_token="<pad>" 
)
tokenizer.model_max_length = _MAX_SEQUENCE_LENGTH


def tokenize_function(examples):
    """Tokenizes input text, ensuring truncation based on the model's max length."""
    return tokenizer(examples["text"], truncation=True) 

# Apply tokenization using multiple processes for efficiency
tokenized_dataset = dataset.map(
    tokenize_function, 
    batched=True, 
    num_proc=os.cpu_count() or 1,
    remove_columns=["text"],
)


def group_texts(examples):
    """
    Groups tokenized texts into contiguous blocks of MAX_SEQUENCE_LENGTH.
    This is essential for Causal Language Modeling training.
    """
    # Concatenate all texts into one list
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Drop the last part if it's shorter than the max length
    total_length = (total_length // _MAX_SEQUENCE_LENGTH) * _MAX_SEQUENCE_LENGTH
    
    # Split into chunks of MAX_SEQUENCE_LENGTH
    result = {
        k: [t[i : i + _MAX_SEQUENCE_LENGTH] for i in range(0, total_length, _MAX_SEQUENCE_LENGTH)]
        for k, t in concatenated_examples.items()
    }
    # For Causal LM, labels are the input tokens shifted (handled by the collator, 
    # but we set them here for consistency)
    result["labels"] = result["input_ids"].copy()
    return result

# Apply the chunking/grouping function
lm_dataset = tokenized_dataset.map(
    group_texts, 
    batched=True, 
    num_proc=os.cpu_count() or 1,
)
train_dataset = lm_dataset["train"]
logger.info(f"Dataset preparation complete. Total training sequences: {len(train_dataset)}")


# --- 5. MODEL CONFIGURATION AND INITIALIZATION ---
# Define a small LlamaConfig to leverage modern improvements (RoPE, RMSNorm, SwiGLU) 
# while keeping the parameter count low for scratch training.
logger.info("Step 3: Initializing Llama model architecture...")
custom_config = LlamaConfig(
    vocab_size=_VOCAB_SIZE,
    hidden_size=512,        # Dimensionality of the embeddings and transformer layers
    intermediate_size=512 * 4, # Dimension of the MLP projection layer (4x hidden_size is standard)
    num_hidden_layers=6,    # Number of transformer layers (6 layers is a small, efficient choice)
    num_attention_heads=8,  # Number of attention heads
    max_position_embeddings=_MAX_SEQUENCE_LENGTH,
    rms_norm_eps=1e-6,      # Epsilon for the RMSNorm layer (Llama standard)
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# Initialize the LlamaForCausalLM model from the custom Config (Full Scratch initialization)
model = LlamaForCausalLM(config=custom_config)
logger.info(f"Model initialized. Total parameters: {model.num_parameters()}")


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
    # Enable mixed-precision training if CUDA is available for speed
    fp16=torch.cuda.is_available(), 
)

# Data Collator: Prepares batches for Causal Language Modeling (mlm=False)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False
)

# Initialize the Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# --- 7. START TRAINING ---
logger.info("Step 4: Starting model training...")
trainer.train()

# --- 8. SAVE FINAL MODEL AND TOKENIZER ---
trainer.save_model(f"./{_MODEL_NAME}_final")
tokenizer.save_pretrained(f"./{_MODEL_NAME}_final")
logger.info("Training complete. Final model and tokenizer saved.")
