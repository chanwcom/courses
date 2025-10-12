# 1. Environment Setup

This section describes how to set up the development environment required for running the project.

---

## 1.1. Create and Activate the Conda Environment

    conda create --name py3_10_hf python=3.10
    conda activate py3_10_hf

---

## 1.2. Install PyTorch

Visit the official PyTorch installation page:  
👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

1. Check your **CUDA version** (e.g., 12.8).  
2. Select the proper command at the bottom of the PyTorch installation table.  
   For example, for CUDA 12.8:

        pip install torch torchvision

Then, install **torchaudio**:

    pip install torchaudio

> 💡 **Note:**  
> `torchaudio` depends on the PyTorch version, so make sure to match their versions.

---

## 1.3. Install Additional Dependencies

Install `soundfile` to enable reading **FLAC audio files**:

    pip install soundfile

Install Hugging Face libraries:

    pip install "transformers[torch]" datasets

Install **WebDataset** for dataset streaming:

    pip install webdataset

Install evaluation utilities for speech recognition:

    pip install evaluate jiwer

---

## 1.4. References

- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)  
- [Hugging Face Datasets Installation](https://huggingface.co/docs/datasets/installation)



# 2. STOP dataset

We used the music portion of the STOP train set.
However, we removed 00011525.wav, since the transcript of it seems to contain an error: "play song TITLE_MEDIA on spotify"
You may download the compressed sharded WebDataset from the following directory:

https://drive.google.com/file/d/1myqysY_FkaynOfkORBA5xyw4FRJ_OxuW/view?usp=drive_link

So the total number of utterances is reduced from 11563 to 11562.

Please note that you should decompress tar.gz files only once. We will use 10 sharded *.tar file for training and eval.

For the test set, I randomly chose 300 utterances from `test_0/music_test`. You may download the compressed sharded WebDataset.
https://drive.google.com/file/d/1j2z8xb4V5zTb6ChJafZZp8Gtt61_ma_1/view?usp=drive_link

As before, you should decompress tar.gz files only once. We will use 10 sharded *.tar file for training and eval.

# 4. Run the scripts in the "run" directory

If GPU0 is available, then set the following configuration variables:
\
`export NCCL_P2P_DISABLE=1; export NCCL_IB_DISABLE=1; export CUDA_VISIBLE_DEVICES=0`
