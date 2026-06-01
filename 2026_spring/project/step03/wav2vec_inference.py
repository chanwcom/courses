# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard imports
import os

# Third-party imports
from transformers import pipeline

# Custom imports
import sample_util_solution
sample_util = sample_util_solution

db_top_dir = # TODO
test_top_dir = os.path.join(#TODO
test_dataset = sample_util.make_dataset(test_top_dir, False)

transcriber = pipeline(
    "automatic-speech-recognition",
    model="/home/chanwcom/local_repositories/cognitive_workflow_kit/tool/models/asr_stop_model_final/checkpoint-2000"
)

# Function to write REF/HYP pairs to a file
def write_results(dataset, transcriber, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for data in dataset:
            ref = data["labels"]
            hyp = transcriber(data["input_values"])["text"]
            f.write(f"REF: {ref}\n")
            f.write(f"HYP: {hyp}\n\n")  # double newline for readability

# Write test_clean_dataset
write_results(test_dataset, transcriber, "test_result.txt")
