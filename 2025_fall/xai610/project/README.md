# 1. Setup the environment.


Create the Conda environment.

`conda create --name py3_10_hf python=3.10`

Activate the Conda environment created just ago.

`conda activate py3_10_hf`


Install Pytorch.

https://pytorch.org/get-started/locally/
Check the CUDA version
\

Select the command at the bottom of the table, after seleting the right "Compute Platform"
For example, if the CUDA version is 11.8, then run the following command:
Note that torchdata is added.


`conda install pytorch torchvision torchaudio torchdata pytorch-cuda=11.8 -c pytorch -c nvidia`
\
`conda install tensorflow-cpu`

Install HuggingFace **Transformers** and **Datasets**.
\
`pip install transformers[torch] datasets`

SoundFile installation
\
`pip install soundfile`

WebDataset installation:
\
`pip install webdataset`

Librosa installtion
\
`conda install -c conda-forge librosa`

For speech recognition evaluation
\
`pip install evaluate jiwer`

Reference:
https://huggingface.co/docs/datasets/v1.11.0/installation.html



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
