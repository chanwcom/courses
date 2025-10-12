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

---

# 3. Running the Scripts in the `run` Directory

If **GPU 0** is available, set the following environment variables before running the scripts:

    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export CUDA_VISIBLE_DEVICES=0

> 💡 **Note:**  
> If a different GPU is available (for example, GPU 1 or GPU 2),  
> replace the value in `CUDA_VISIBLE_DEVICES` with the corresponding GPU ID.  
> For example:
>
>     export CUDA_VISIBLE_DEVICES=1
>
> This ensures that your training script runs on the correct GPU device.



 'eval_steps_per_second': 3.887, 'epoch': 211.0}
{'loss': 0.0136, 'grad_norm': 0.2718251645565033, 'learning_rate': 5.066666666666667e-06, 'epoch': 213.0}                     
{'loss': 0.0142, 'grad_norm': 0.5199770331382751, 'learning_rate': 3.4000000000000005e-06, 'epoch': 216.0}                    
{'loss': 0.0128, 'grad_norm': 0.5341504216194153, 'learning_rate': 1.7333333333333334e-06, 'epoch': 219.0}                    
{'loss': 0.013, 'grad_norm': 0.33202001452445984, 'learning_rate': 6.666666666666667e-08, 'epoch': 222.0}                     
{'eval_loss': 0.5652906894683838, 'eval_wer': 0.22662431527693244, 'eval_runtime': 28.2358, 'eval_samples_per_second': 92.79, 'eval_steps_per_second': 3.896, 'epoch': 222.0}
{'train_runtime': 1881.837, 'train_samples_per_second': 34.009, 'train_steps_per_second': 1.063, 'train_loss': 1.1427945377230644, 'epoch': 222.0}
100%|█████████████████████████████████████████████████████████████████████████████████████| 2000/2000 [31:21<00:00,  1.06it/s]
