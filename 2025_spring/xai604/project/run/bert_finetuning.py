# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import glob
import os
from typing import Dict

# Third-party imports
import torch
import evaluate
import numpy as np
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils import data

# Custom imports
import sample_util

model_name = "google-bert/bert-base-uncased"
db_top_dir = "/mnt/data/database"
train_top_dir = os.path.join(db_top_dir, "stop_music/music_train")
test_top_dir = os.path.join(db_top_dir, "stop_music/music_test0")
output_dir = "/mnt/data/home/chanwcom/experiment/bert_stop_model_final"

INTENTS = [
    "ADD_TO_PLAYLIST_MUSIC",
    "CREATE_PLAYLIST_MUSIC",
    "DISLIKE_MUSIC",
    "LIKE_MUSIC",
    "LOOP_MUSIC",
    "PAUSE_MUSIC",
    "PLAY_MUSIC",
    "PREVIOUS_TRACK_MUSIC",
    "REMOVE_FROM_PLAYLIST_MUSIC",
    "REPLAY_MUSIC",
    "SKIP_TRACK_MUSIC",
    "START_SHUFFLE_MUSIC",
    "STOP_MUSIC",
    "UNSUPPORTED_MUSIC",
    "SET_DEFAULT_PROVIDER_MUSIC",
]

INTENT2IDX = {intent: idx for idx, intent in enumerate(INTENTS)}

def find_index(intent: str) -> int:
    assert intent in INTENT2IDX, f"Unknown intent: {intent}"
    return INTENT2IDX[intent]


tokenizer = AutoTokenizer.from_pretrained(model_name)
def tokenize_function(examples):
    return tokenizer([examples], padding="max_length", truncation=True)

def preprocess_examples(sample: Dict) -> Dict:
    output = {}
    output["label"] = find_index(sample["meta"]["intents"][0]["intent"])
    output["text"] = sample["labels"]
    tokenized = tokenize_function(output["text"])
    for key in tokenized.keys():
        output[key] = tokenized[key][0]

    return output


train_dataset = (sample_util.make_dataset(train_top_dir, do_tokenize=False)
    .map(lambda example: preprocess_examples(example)))
test_dataset = (sample_util.make_dataset(test_top_dir, do_tokenize=False)
    .map(lambda example: preprocess_examples(example)))

model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                           num_labels=15)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    # Directory to save model checkpoints and outputs.
    output_dir=output_dir,
    per_device_train_batch_size=40,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=200,
    max_steps=400,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    per_device_eval_batch_size=40,
    save_steps=400,
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
