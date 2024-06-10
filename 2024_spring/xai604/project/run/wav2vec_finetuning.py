# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import glob
import os

# Third-party imports
from transformers import AutoModelForCTC, TrainingArguments, Trainer
from datasets import load_dataset, Audio
from transformers import AutoProcessor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from torch.utils import data
import tensorflow as tf
import torch
import evaluate
import numpy as np

# Custom imports
from data import speech_data_helper
from typing import Any, Dict, List, Optional, Union

db_top_dir = "/home/chanwcom/databases/"
#db_top_dir = "/home/chanwcom/speech_database"
train_top_dir = os.path.join(db_top_dir, "stop/music_train_tfrecord")
test_top_dir = os.path.join(db_top_dir,
                            "stop/test_0_music_random_300_tfrecord")

# yapf: disable
op = speech_data_helper.SpeechDataToWave()
train_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(train_top_dir, "*tfrecord-*")),
              compression_type="GZIP")
train_dataset = train_dataset.batch(1)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(op.process)
# yapf: enable

# yapf: disable
test_dataset = tf.data.TFRecordDataset(
    glob.glob(os.path.join(test_top_dir, "*tfrecord-*")),
              compression_type="GZIP")
test_dataset = test_dataset.batch(1)
test_dataset = test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(op.process)
# yapf: enable


def to_torch(inputs: dict):
    for key in inputs.keys():
        inputs[key] = torch.tensor(inputs[key].numpy())

    return inputs

processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

class IterDataset(data.IterableDataset):

    def __init__(self, tf_dataset):
        self._dataset = tf_dataset
        op = speech_data_helper.SpeechDataToWave()

        # The following line is neede, otherwise ..dataset.map will not work

        # Parses the serialized data.
        #self._dataset = self._dataset.map(op.process)

    def __iter__(self):
        for data in self._dataset:
            output = {}
            output["input_values"] = [tf.squeeze(data[0]["SEQ_DATA"]).numpy()]
            output["input_length"] = tf.squeeze(data[0]["SEQ_LEN"]).numpy()
            with processor.as_target_processor():
                output["labels"] = processor(
                    data[1]["SEQ_DATA"][0].numpy().decode(
                        "unicode_escape")).input_ids

            yield (output)


pytorch_train_dataset = IterDataset(train_dataset)
pytorch_test_dataset = IterDataset(test_dataset)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = evaluate.load("wer")
    wer = wer.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


@dataclass
class DataCollatorCTCWithPadding:
    processor: AutoProcessor
    padding: Union[bool, str] = "longest"

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{
            "input_values": feature["input_values"][0]
        } for feature in features]
        label_features = [{
            "input_ids": feature["labels"]
        } for feature in features]

        batch = self.processor.pad(input_features,
                                   padding=self.padding,
                                   return_tensors="pt")

        labels_batch = self.processor.pad(labels=label_features,
                                          padding=self.padding,
                                          return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor,
                                           padding="longest")

model = AutoModelForCTC.from_pretrained(
    "facebook/wav2vec2-base",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id)

# asr_mind_model00: 16 batch, 1e-4
# asr_mind_model01: 16 batch, 3e-4

# asr_stop_model_00: 100h, 1e-4, 32batch   7.3% WER?
# asr_stop_model_01: base, 1e-4, 32batch
# asr_stop_model_02: base, 2e-4, 40batch, 5000
# asr_stop_model_03: base, 1e-4, 40batch, 5000
# asr_stop_model_04: base, 1e-4, 40batch, 10000
# asr_stop_model_05: base, 1e-4, 40batch, 10000, sum
# asr_stop_model_06: base, 1e-4, 40batch, 10000, grad_acc = 1

training_args = TrainingArguments(
    output_dir=
    "/home/chanwcom/local_repositories/cognitive_workflow_kit/tool/models/asr_stop_model_final",
    per_device_train_batch_size=40,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=10000,
    gradient_checkpointing=True,
    fp16=True,
    #group_by_length=True,
    eval_strategy="steps",
    per_device_eval_batch_size=40,
    save_steps=5000,
    eval_steps=100,
    logging_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=pytorch_train_dataset,
    eval_dataset=pytorch_test_dataset,
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
