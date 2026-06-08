# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import argparse
import os
import re
import itertools
import shutil

# Third-party imports
import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from transformers import AutoProcessor, PreTrainedTokenizer

# Custom imports
import sample_util

# TODO: Correct paths depending on your environment
db_top_dir = ""
train_top_dir = os.path.join(db_top_dir, "libri_light/1h")
test_top_dir = os.path.join(db_top_dir, "test-clean")
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

train_dataset = sample_util.make_dataset(train_top_dir)

model_top_dir = ("")
spm_top_dir = ("")
# End of ToDO

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

class Wav2Vec2SPMTokenizer(PreTrainedTokenizer):
    """Custom Tokenizer for Wav2Vec2 using SentencePiece.

    Inherits from PreTrainedTokenizer to avoid the mandatory vocab.json 
    requirement of Wav2Vec2CTCTokenizer.
    """

    def __init__(self, spm_model_path: str, **kwargs: Any):
        """Initializes the tokenizer and loads the SentencePiece model."""
        import sentencepiece as spm
        self.spm_model_path = spm_model_path
        self.sp = spm.SentencePieceProcessor(model_file=spm_model_path)
        
        # Standard CTC special tokens are passed to the base class.
        super().__init__(
            pad_token="<pad>",
            unk_token="<unk>",
            bos_token="<s>",
            eos_token="</s>",
            **kwargs
        )

    @property
    def vocab_size(self) -> int:
        """Returns the size of the SentencePiece vocabulary."""
        return self.sp.get_piece_size()

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary as a dictionary for compatibility."""
        return {
            self.sp.id_to_piece(i): i for i in range(self.vocab_size)
        }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes text using the SentencePiece engine."""
        return self.sp.encode_as_pieces(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Converts a subword piece to its integer ID."""
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Converts an integer ID to its subword piece."""
        return self.sp.id_to_piece(index)

    def _decode(self, 
                token_ids: List[int], 
                group_tokens: bool = True, 
                **kwargs: Any) -> str:
        """Decodes IDs with CTC collapse and SentencePiece."""
        if group_tokens:
            token_ids = [k for k, _ in itertools.groupby(token_ids)]
        
        # Remove padding and ignore index (-100).
        filtered_ids = [
            int(i) for i in token_ids 
            if i != self.pad_token_id and i != -100
        ]
        return self.sp.decode(filtered_ids) if filtered_ids else ""

    def save_vocabulary(self, 
                        save_directory: str, 
                        filename_prefix: Optional[str] = None) -> tuple:
        """Saves the SPM model file. Fixes the NotImplementedError."""
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)

        file_name = "tokenizer.model"
        if filename_prefix:
            file_name = f"{filename_prefix}-{file_name}"
            
        vocab_file = os.path.join(save_directory, file_name)
        
        # Copy the original .model file to the checkpoint directory.
        if os.path.abspath(self.spm_model_path) != os.path.abspath(vocab_file):
            shutil.copyfile(self.spm_model_path, vocab_file)
            
        return (vocab_file,)

def compute_metrics(pred) -> Dict[str, float]:
    """Compute Word Error Rate (WER) by mapping sub-labels back to vocab.

    boundary_id and padding are ignored.

    Args:
        pred: A prediction object containing:
            - predictions: Logits of shape (batch, seq, vocab * factor).
            - label_ids: Ground truth IDs (batch, seq).

    Returns:
        A dictionary containing the 'wer' score.
    """
    pred_logits = pred.predictions
    # Get the best token IDs from the extended vocabulary space.
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Use the original SPM vocab size for the modulo operation.
    actual_vocab_size = processor.tokenizer.vocab_size + 1
    boundary_id = actual_vocab_size - 1

    # Map extended sub-label IDs back to the original vocabulary range.
    pred_ids = pred_ids % actual_vocab_size
    pred_ids[pred_ids == boundary_id] = processor.tokenizer.pad_token_id

    # Prepare labels: map back to original range and handle ignore index.
    label_ids = pred.label_ids.copy()
    valid_label_mask = (label_ids != -100)
    label_ids[valid_label_mask] = label_ids[valid_label_mask] % actual_vocab_size
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_ids[label_ids == boundary_id] = processor.tokenizer.pad_token_id

    def clean_special_tokens(text: str) -> str:
        """Removes start/end-of-sentence markers and extra whitespace."""
        text = re.sub(r'^<s>\s*', '', text)
        text = re.sub(r'\s*</s>$', '', text)
        return text.strip()

    # Decode predictions (CTC grouping enabled).
    pred_str = processor.batch_decode(
        pred_ids,
        group_tokens=True,
        skip_special_tokens=False,
    )

    # Decode ground truth labels.
    label_str = processor.batch_decode(
        label_ids,
        group_tokens=False,
        skip_special_tokens=False
    )

    # Clean and calculate WER.
    pred_str = [clean_special_tokens(s) for s in pred_str]
    label_str = [clean_special_tokens(s) for s in label_str]

    wer_metric = evaluate.load("wer")
    wer_score = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer_score}



@dataclass
class DataCollatorCTCWithPadding:
    """Data collator that dynamically pads inputs and labels for CTC training.

    This class pads the input audio features and the corresponding label
    sequences to the length of the longest element in the batch. It also
    replaces padding tokens in the labels with -100.

    Attributes:
        processor (AutoProcessor): The processor used for feature extraction and tokenization.
        padding (Union[bool, str]): Padding strategy. Defaults to "longest" to pad to the
            longest sequence in the batch.
    """

    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Pad inputs and labels in a batch for model training.

        Args:
            features: A list of feature dictionaries, each containing:
                - "input_values": the audio features (list or tensor).
                - "labels": the tokenized label sequence.

        Returns:
            A dictionary with padded input tensors and labels ready for the model:
            - "input_values": Padded input audio feature tensor.
            - "labels": Padded label tensor with padding tokens replaced by -100.
        """
        input_features = [{"input_values": feature["input_values"]}
                          for feature in features]
        label_features = [{"input_ids": feature["labels"]}
                          for feature in features]
        # Use the processor's pad method to pad input audio features to the same length.
        # Without return_attention_mask, Wav2Vec2 does not generate the mask.
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
            return_attention_mask=True
        )

        # Pad the label sequences separately using the processor's pad method.
        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace padding tokens in labels with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Add the processed labels to the batch dictionary.
        batch["labels"] = labels
        return batch

# Define the training arguments for the Hugging Face Trainer.
# These control training hyperparameters and runtime behavior:
training_args = TrainingArguments(
    # Directory to save model checkpoints and outputs.
    output_dir="/home/chanwcom/local_repositories/cognitive_workflow_kit/tool/"
               "models/asr_stop_model_final",

    # Batch size per device (GPU/CPU) for training.
    per_device_train_batch_size=24,

    # Number of batches to accumulate gradients over before updating model weights.
    gradient_accumulation_steps=2,

    # Initial learning rate for the optimizer.
    learning_rate=1e-4,

    # Number of warmup steps to gradually increase learning rate at start.
    warmup_steps=500,

    # Total number of training steps.
    max_steps=2000,

    # Enable gradient checkpointing to reduce memory usage at the cost of extra compute.
    gradient_checkpointing=True,

    # Use mixed precision training (float16) to speed up training and reduce memory.
    fp16=True,

    # Performs evaluation every N steps (eval_strategy="steps").
    eval_strategy="steps",

    # Batch size per device during evaluation.
    per_device_eval_batch_size=24,

    # Save model checkpoints every N steps.
    save_steps=2000,

    # Run evaluation every N steps during training.
    eval_steps=100,

    # Log training progress every N steps.
    logging_steps=25,

    # Load the best model (lowest WER) at the end of training automatically.
    load_best_model_at_end=True,

    # Metric to use for selecting the best model checkpoint.
    metric_for_best_model="wer",

    # Indicates that a lower metric score (WER) is better.
    greater_is_better=False,

    # Disable pushing model to the Hugging Face hub.
    push_to_hub=False,
)




def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Wav2Vec2 Training with Dynamic Vocab Size")
    parser.add_argument("--vocab_size", type=int, default=None,
                        help="Vocabulary size (e.g., 32, 128).")
    return parser.parse_args()


def main():
    args = parse_args()

    global processor

    # Dynamic configuration based on vocab_size
    if args.vocab_size is not None:
        spm_name = f"librispeech_unigram_{args.vocab_size}.model"
        spm_model_path = os.path.join(spm_top_dir, spm_name)
        current_vocab_size = args.vocab_size
        out_name = f"shc_2000_steps_unigram_{args.vocab_size}_00"

        # Inject the SPM wrapper into the existing processor structure
        processor.tokenizer = Wav2Vec2SPMTokenizer(spm_model_path)

    else:
        spm_model_path = None
        current_vocab_size = 32
        out_name = "ctc_2000_steps_default_vocab_01"

    current_vocab_size += 1

    # Dataset preparation
    train_dataset = sample_util.make_dataset(
        train_top_dir, True, spm_model_path)
    test_dataset = sample_util.make_dataset(
        test_top_dir, True, spm_model_path)

    # Initialize data collator
    data_collator = DataCollatorCTCWithPadding(
        processor=processor, padding="longest")

    # Load model with dynamic vocab size
    model = AutoModelForCTC.from_pretrained(
        "facebook/wav2vec2-base",
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=current_vocab_size,
        ignore_mismatched_sizes=True
    )


    # TODO
    # Create the Trainer instance to handle training and evaluation.
    # This ties together the model, datasets, tokenizer, data collator, and metrics.
    trainer = Trainer(
        model=,
        args=,
        train_dataset=,
        eval_dataset=,
        tokenizer=,
        data_collator=,
        compute_metrics=,
        vocab_size=,
    )
    # End of TODO

    trainer.train()


if __name__ == "__main__":
    main()
