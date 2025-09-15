# pylint: disable=import-error, no-member
from __future__ import (absolute_import, division, print_function,
                         unicode_literals)

__author__ = "Chanwoo Kim(chanwcom@gmail.com)"

# Standard library imports
import glob
import io
import os
from typing import Dict

# Third-party imports
import json
import torchaudio
import webdataset as wds
from transformers import AutoProcessor

# Define processor globally (assumed to be initialized elsewhere in actual code)
processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base")

def preprocess_sample(sample: Dict, do_tokenize: bool = True) -> Dict:
    """Preprocess a single raw sample from the WebDataset.

    This function loads the waveform from the raw bytes using torchaudio,
    extracts features using the processor's feature extractor, and tokenizes
    the transcript text.

    Args:
        sample (Dict): A dictionary containing keys 'wav' (raw audio bytes)
            and 'txt' (transcript bytes).
        do_tokenize (bool): Whether to tokenize the transcript text.

    Returns:
        Dict: A dictionary with keys:
            - "input_values": processed audio feature tensor.
            - "labels": list of token IDs corresponding to the transcript.
            - "meta": A meta data information parsed from json field.
    """
    waveform, sample_rate = torchaudio.load(io.BytesIO(sample["wav"]))
    input_values = processor.feature_extractor(
        waveform[0], sampling_rate=sample_rate
    ).input_values[0]

    result = {
        "input_values": input_values,
        "meta":  sample["meta"]
    }

    text = sample["txt"]
    if do_tokenize:
        labels = processor.tokenizer(text).input_ids
        result["labels"] = processor.tokenizer(text).input_ids
    else:
        result["labels"] = text

    return result

def make_dataset(data_dir: str, do_tokenize: bool = True) -> wds.WebDataset:
    """Creates a WebDataset pipeline that loads and preprocesses data shards.

    It reads all shards named 'shard-*.tar' in the given directory,
    extracts 'wav' and 'txt' entries as tuples, converts them into dictionaries,
    and applies the preprocessing function.

    Args:
        data_dir (str): Path to the directory containing dataset shards.
        do_tokenize (bool): Whether to tokenize the transcript text.

    Returns:
        wds.WebDataset: The prepared dataset pipeline with preprocessing.
    """
    dataset = (
        wds.WebDataset(glob.glob(os.path.join(data_dir, "shard-*.tar")))
        .to_tuple("wav", "txt", "json")
        .map(lambda sample:
             {"wav": sample[0],
              "txt": sample[1].upper().decode("utf-8"),
              "meta": json.loads(sample[2].decode("utf-8"))})
        .map(lambda sample: preprocess_sample(sample, do_tokenize=do_tokenize))
    )
    return dataset
