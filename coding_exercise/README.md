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


---

## 2. Libri-Light 1-Hour Fine-Tuning Set and Test-Clean and Test-Other Sets

We use the **1-hour Libri-Light fine-tuning set** for our experiments. This dataset is particularly suitable for low-resource ASR fine-tuning.

**Download link:**  
[Libri-Light 1-hour fine-tuning set](https://drive.google.com/drive/folders/1izfwIUAreziLLpLUCAl7_zh10LzxxIKg?usp=drive_link)

**Important Notes:**

- Decompress each `.tar.gz` file **only once**.
- For training and evaluation, we use **5 sharded `.tar` files**.
- For testing, we use the **LibriSpeech `test-clean` and `test-other`** sets.

This dataset is a subset of the **Libri-Light** corpus, introduced by Kahn et al., 2020, designed for **self-supervised and semi-supervised speech representation learning**. The 1-hour fine-tuning set allows quick adaptation experiments in low-resource settings.

**Reference:**  
Kahn, J., et al. (2020). *Libri-Light: A Benchmark for ASR with Limited or No Supervision*. [arXiv:2009.08568](https://arxiv.org/abs/2009.08568)

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
