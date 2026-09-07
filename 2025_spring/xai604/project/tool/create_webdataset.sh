#!/bin/bash

#python create_stop_webdataset.py \
#    --audio_dir=/mnt/nas2dual/database/stop_decompressed/stop/train/music_train_flac \
#    --schema_file=../resources/top.schema.json \
#    --output_dir=/mnt/kioxia_exeria/home/chanwcom/stop_database/music_train \
#    --min_shard_count=10

python create_stop_webdataset.py \
    --audio_dir=/mnt/nas2dual/database/stop_decompressed/stop/test_0/music_test_flac \
    --schema_file=../resources/top.schema.json \
    --output_dir=/mnt/kioxia_exeria/home/chanwcom/stop_database/music_test0 \
    --min_shard_count=10
