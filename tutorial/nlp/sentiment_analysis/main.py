"""A module to fine-tune a DistilBERT model for sentiment analysis.

This script demonstrates a minimal Hugging Face pipeline using the IMDB
dataset. It follows Google's 80-character width and docstring standards.
"""

import numpy as np
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def preprocess_data(examples, tokenizer):
    """Tokenizes text data with padding and truncation for model input.

    Args:
        examples: A dictionary containing 'text' fields from the dataset.
        tokenizer: A Hugging Face tokenizer instance.

    Returns:
        A dictionary of tokenized features (input_ids, attention_mask).
    """
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128  # Keep max_length short for faster training.
    )


def main():
    """Executes the fine-tuning pipeline on a subset of the IMDB dataset."""
    model_id = "distilbert-base-uncased"
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Use a small subset (1k samples) to ensure quick execution.
    # We shuffle with a fixed seed for reproducibility.
    train_set = dataset["train"].shuffle(seed=42).select(range(1000))
    eval_set = dataset["test"].shuffle(seed=42).select(range(1000))

    # Apply preprocessing using a lambda to pass the tokenizer.
    train_set = train_set.map(
        lambda x: preprocess_data(x, tokenizer), batched=True
    )
    eval_set = eval_set.map(
        lambda x: preprocess_data(x, tokenizer), batched=True
    )

    # Initialize model for binary classification (Positive/Negative).
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=2
    )

    # Define training arguments following 80-char formatting.
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
    )

    # Start the fine-tuning process.
    trainer.train()


if __name__ == "__main__":
    main()
