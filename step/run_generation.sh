#!/bin/bash

set -o errexit
set -o nounset
set -o xtrace

source /mnt/nlp_swordfish/storage/proj/zj2265/venv/bin/activate
export PYTHONPATH=/mnt/nlp_swordfish/storage/proj/zj2265/MLMGen
export CUDA_VISIBLE_DEVICES=0,1,2,3


SAMPLE_PER_ITEM=10
CONFIG_FILE_PATH=../yamls/examine_hidden_states.yaml
DEVICE=cuda:1

python3 generate_sentence_batches.py --num_samples_per_item ${SAMPLE_PER_ITEM}\
    --device DEVICE --config_file CONFIG_FILE_PATH
