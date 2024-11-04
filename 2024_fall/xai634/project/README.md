# 1. Setup the environment.


Create the Conda environment.

`conda create --name py3_10_hf python=3.10`

Activate the Conda environment created just ago.

`conda activate py3_10_hf`


Install Pytorch and Tensorflow.

https://pytorch.org/get-started/locally/
Check the CUDA version
\
`nvcc --version`

Select the command at the bottom of the table, after seleting the right "Compute Platform"
For example, if the CUDA version is 11.8, then run the following command:
Note that torchdata is added.


`conda install pytorch torchvision torchaudio torchdata pytorch-cuda=11.8 -c pytorch -c nvidia`
\
`conda install tensorflow-cpu`

Install HuggingFace **Transformers** and **Datasets**.
\
`pip install transformers datasets`

SoundFile installation
\
`pip install soundfile`

Librosa installtion
\
`conda install -c conda-forge librosa`

For speech recognition evaluation
\
`pip install evaluate jiwer`

Reference:
https://huggingface.co/docs/datasets/v1.11.0/installation.html

# 2. Bazel and protoc installation

Use bazelisk
https://bazel.build/install/ubuntu

`sudo apt install protobuf-compiler`


# 3. STOP dataset

We used the music portion.
But we removed 00011525.wav.
You may download the compressed sharded TFRecord from the following directory:
https://drive.google.com/file/d/1FUrwZzeZ8S1su9MPaQVu4WDswM2AjPxG/view?usp=drive_link

So the total number of utterances is reduced from 11563 to 11562.

TFRecord can be found here.
https://drive.google.com/file/d/1FUrwZzeZ8S1su9MPaQVu4WDswM2AjPxG/view?usp=drive_link

# 4. Run the scripts in the "run" directory

If GPU0 is available, then set the following configuration variables:
\
`export NCCL_P2P_DISABLE=1; export NCCL_IB_DISABLE=1; export CUDA_VISIBLE_DEVICES=0`


`bazel run :wav2vec_training`
\
`bazel run :wav2vec_inference`
\
`bazel run :bert_training`
\
`bazel run :bert_inference`
