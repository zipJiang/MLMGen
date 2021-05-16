#!/bin/bash

set -o errexit
set -o nounset
set -o xtrace

# source /mnt/nlp_swordfish/storage/proj/zj2265/venv/bin/activate
export PYTHONPATH=/home/zipjiang/Research/Columbia/MLMGen
export CUDA_VISIBLE_DEVICES=0


SAMPLE_PER_ITEM=100
CONFIG_FILE_PATH=../yamls/examine_hidden_states.yaml
DEVICE=cuda

python3 generate_sentence_batches.py --num_samples_per_item ${SAMPLE_PER_ITEM}\
    --device ${DEVICE} --config_file ${CONFIG_FILE_PATH}
