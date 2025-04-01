#!/bin/bash
set -eu  # Exit on error

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

storage_dir=
librispeech_dir=$storage_dir/LibriSpeech
wham_dir=$storage_dir/wham_noise
local_dir=
output_dir=$local_dir/output
test_base_dir=../../    # espnet/egs2/librimix/ehn1


# Path to python
python_path=

modes="min"
subset_names="test"

# noise name map
declare -A ns_subset_name_map
ns_subset_name_map=(["train"]="tr" ["test"]="tt" ["dev"]="cv")

# librispeech name map
declare -A ls_subset_name_map
ls_subset_name_map=(["train"]="train-clean-100" ["test"]="test-clean" ["dev"]="dev-clean")

#         min   max
# train  -2.5  -2.5
# test   -3.5  -3.5
# dev    -3.5  -3.5

declare -A snr_map
snr_map=(["train"]="-2.5" ["test"]="-3.5" ["dev"]="-3.5")

declare -A target_num
target_num=(["train"]=18000 ["test"]=10000 ["dev"]=3000)

mkdir -p $output_dir

stage=1
stop_stage=2

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "Stage 1: Data Generation"
    cd $local_dir
    for subset_name in $subset_names; do
        for mode in $modes; do
            log "Generating data for $subset_name."
            snr=${snr_map[$subset_name]}

            # generate wav data
            log "Generating wav data."
            $python_path create_s2m.py --target_num ${target_num[$subset_name]} \
                --ls_subset_name ${ls_subset_name_map[$subset_name]} \
                --hs_subset_name $subset_name \
                --ns_subset_name ${ns_subset_name_map[$subset_name]} \
                --output_dir $output_dir/$subset_name \
                --snr $snr \
                --modes $mode
        done
    done
    log "Stage 1 done"
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "Stage 2: Evaluation"
    cd $test_base_dir
    for subset_name in $subset_names; do
        for mode in $modes; do
            snr=${snr_map[$subset_name]}

            log "Evaluating \`$subset_name $mode $snr\`."
            rm -rf $test_base_dir/my_dump/raw/*
            mkdir -p $test_base_dir/my_dump/raw/scps
            cp $output_dir/$subset_name/wav16k/$mode/scps/*.scp $test_base_dir/my_dump/raw/scps
            bash run_eval_s2m.sh $mode
            cp -r $test_base_dir/my_dump/raw/scps/scoring $output_dir/$subset_name/wav16k/$mode
            awk 'NR > 0 { sum += $2; count++ } END { print "spk1 Average:", sum/count }' $output_dir/$subset_name/wav16k/$mode/scoring/SI_SNR_spk1 \
                >> $output_dir/$subset_name/wav16k/$mode/scoring/avg.md
            awk 'NR > 0 { sum += $2; count++ } END { print "spk2 Average:", sum/count }' $output_dir/$subset_name/wav16k/$mode/scoring/SI_SNR_spk2 \
                >> $output_dir/$subset_name/wav16k/$mode/scoring/avg.md
            mkdir $output_dir/$subset_name/wav16k/$mode/results
            mv $output_dir/$subset_name/wav16k/$mode/scoring/avg.md $output_dir/$subset_name/wav16k/$mode/results
            cp $test_base_dir/my_dump/raw/RESULTS.md $output_dir/$subset_name/wav16k/$mode/results
            log "Evaluation of \`$subset_name $mode $snr\` done."
        done
    done
    log "Stage 2 done"
fi
