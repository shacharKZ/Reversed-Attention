# bin/bash

# DEVICE=$1
# DS=$2

DEVICE=0
DS="antonym_len_1"

echo $DS

for MODEL in "facebook/opt-1.3b" "gpt2-xl"
do
    echo $MODEL
    python attention_patching.py --model_name $MODEL --dataset_name $DS --out_folder  intervention_plot1 --device "cuda:$DEVICE" --check_if_result_already_exists
done

