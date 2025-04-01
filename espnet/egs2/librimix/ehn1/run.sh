#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

./enh_kws.sh --use_noise_ref false --train_set train-snips-min \
    --valid_set dev-snips-min --test_sets 'test-snips-min test-min' \
    --fs 16k --audio_format wav --local_data_opts '--sample_rate 16k --min_or_max min' \
    --lang en --kws_enh_task true \
    --ngpu 1 --num_nodes 1 \
    --enh_config conf/train_tskim.yaml --enh_exp exp/tskim-min-concat_avgpooling \
    --stage 5 --stop_stage 8 "$@"; exit $?
