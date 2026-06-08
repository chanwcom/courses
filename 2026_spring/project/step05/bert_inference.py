# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import glob
import os
from typing import Dict

# Third-party imports
from transformers import pipeline
from transformers import AutoTokenizer
from torch.utils import data
import torch
import evaluate

# Custom imports
import sample_util

# TODO: Correct paths depending on your environment
model_name = ""
db_top_dir = ""
train_top_dir = os.path.join(db_top_dir, "stop_music/music_train")
test_top_dir = os.path.join(db_top_dir, "stop_music/music_test0")
output_dir = ""
# End of TODO


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

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
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

test_dataset = (sample_util.make_dataset(test_top_dir, do_tokenize=False)
    .map(lambda example: preprocess_examples(example)))

metric = evaluate.load("accuracy")

# TODO: Construct the pipeline
classifier = pipeline(
    "text-classification",
    model=,
    tokenizer=)
# End of TODO

count = 0
correct = 0
for data in test_dataset:
    # TODO: Complete the following portions to check the number of correctly 
    # classified cases.
    ref = 
    hyp = 
    hyp_index = 

    print(f"REF: {ref}    HYP: {hyp_index}")
    count += 1
    correct += (ref == hyp_index)
    # End of TODO

print ("===============")
print(f"Accuracy: {correct / count}")
print ("===============")
