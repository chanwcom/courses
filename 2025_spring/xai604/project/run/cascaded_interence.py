# pylint: disable=import-error, no-member

from __future__ import absolute_import, division, print_function, unicode_literals

__author__ = "Chanwoo Kim (chanwcom@gmail.com)"

# Standard library imports
import os

# Third-party library imports
from transformers import pipeline
from transformers import AutoTokenizer


# Local application imports
import sample_util

# Defines the path to the root directory containing the dataset.
db_top_dir = "/mnt/data/database"

# Defines the model names.
asr_model_name = ("/mnt/data/home/chanwcom/experiment/wav2vec2_stop_model_final/"
              "checkpoint-2000")
bert_model_name = ("/mnt/data/home/chanwcom/experiment/bert_stop_model_final/"
              "checkpoint-400")
bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
def bert_tokenize_function(examples):
    return bert_tokenizer([examples], padding="max_length", truncation=True)



# Defines the path to the test dataset.
test_top_dir = os.path.join(db_top_dir, "stop_music/music_test0")

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


class Module(object):
    def __init__(self, task: str, model_name: str, **kwargs):
        # Initializes the ASR pipeline with a fine-tuned checkpoint.
        # This loads a model for automatic speech recognition (ASR).
        self._pipeline = pipeline(
            task,
            model=model_name,
            **kwargs
        )
        pass

    def run(self, inputs):
        return self._pipeline(inputs)

# Creates the ASR and BERT modules.
asr_module = Module("automatic-speech-recognition", asr_model_name)
bert_module = Module(
    "text-classification", bert_model_name, tokenizer=bert_tokenizer)


# Loads the test dataset using a custom utility function.
# The second argument (False) disables tokenization inside the dataset.
test_dataset = (sample_util.make_dataset(test_top_dir, do_tokenize=False)
                .map(lambda examples:
                     {"text": asr_module.run(examples["input_values"])["text"],
                      "meta": examples["meta"]})
                .map(lambda examples:
                     {"hyp": bert_module.run(examples["text"]),
                      "meta": examples["meta"]}))

count = 0
correct = 0
# Iterate through each sample in the dataset and decode the audio
for data in test_dataset:
    # Ground-truth reference text (original transcript).
    ref_index = find_index(data["meta"]["intents"][0]["intent"])

    # Inference output from the cascaded ASR/BERT models.
    hyp_index = int(data['hyp'][0]['label'].split("_")[1])

    # Print the reference and hypothesis for comparison
    print(f"REF: {ref_index}    HYP: {hyp_index}")

    count += 1
    correct += (ref_index == hyp_index)

print ("===============")
print(f"Accuracy: {correct / count}")
print ("===============")
