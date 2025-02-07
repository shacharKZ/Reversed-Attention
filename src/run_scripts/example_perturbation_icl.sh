#!/bin/sh

# DEVICE=$1
DEVICE=0

for DS in "antonym" "alphabetically_first_3" "next_item" "present-past"
do
    for MODEL in "facebook/opt-1.3b" "gpt2-xl"
    do
        echo $DS
        echo $MODEL
        echo "DEVICE: cuda:$DEVICE"

        TMP_MODEL=$(echo $MODEL | tr / -)

        CM_FOLDERS="results_$TMP_MODEL"
        ATTN_MAPS_FOLDER="results_v1"
        PERTURBATION_FOLDER="results_v1"

        python perturbation_exp.py --model_name $MODEL --dataset_name $DS --out_folder $PERTURBATION_FOLDER --heads_list_path rand --device "cuda:$DEVICE" --check_if_result_already_exists
        
        python perturbation_exp.py --model_name $MODEL --dataset_name $DS --out_folder $PERTURBATION_FOLDER --heads_list_path index --device "cuda:$DEVICE" --check_if_result_already_exists

        for N_SHOTS in 0 1 5 10
        do
            # run the code from the function_vectors/src folder
            cd ../function_vectors/src
            # LTO: Last Token Only
            python compute_indirect_effect.py --dataset_name $DS --model_name $MODEL --save_path_root "../$CM_FOLDERS/LTO$N_SHOTS" --n_shots $N_SHOTS --device "cuda:$DEVICE" --last_token_only True
            # go back to the main src folder
            cd ../../src
            CURR_HEADS_PATH="../function_vectors/$CM_FOLDERS/LTO$N_SHOTS/$DS/"$DS"_indirect_effect.pt"
            echo $CURR_HEADS_PATH
            python perturbation_exp.py --model_name $MODEL --dataset_name $DS --out_folder $PERTURBATION_FOLDER --heads_list_path $CURR_HEADS_PATH --device "cuda:$DEVICE" --check_if_result_already_exists


            python causal_mediation_computation.py --model_name $MODEL --dataset_name $DS --n_shots_icl $N_SHOTS --out_folder $ATTN_MAPS_FOLDER --device "cuda:$DEVICE" --check_if_result_already_exists
            CURR_HEADS_PATH="$ATTN_MAPS_FOLDER/cm1_"$TMP_MODEL"_"$DS"_[$N_SHOTS]___simple_causal_mediation.pt"
            echo $CURR_HEADS_PATH
            python perturbation_exp.py --model_name $MODEL --dataset_name $DS --out_folder $PERTURBATION_FOLDER --heads_list_path $CURR_HEADS_PATH --device "cuda:$DEVICE" --check_if_result_already_exists


            echo "create the forward and backward attention maps"
            python attention_maps_computation.py --model_name $MODEL --dataset_name $DS --n_shots_icl $N_SHOTS --out_folder $ATTN_MAPS_FOLDER --device "cuda:$DEVICE" --check_if_result_already_exists
            
            CURR_HEADS_PATH="$ATTN_MAPS_FOLDER/av1_"$TMP_MODEL"_"$DS"_[$N_SHOTS]___backward_attn_maps_norms_mean.csv"
            echo $CURR_HEADS_PATH
            python perturbation_exp.py --model_name $MODEL --dataset_name $DS --out_folder $PERTURBATION_FOLDER --heads_list_path $CURR_HEADS_PATH --device "cuda:$DEVICE" --check_if_result_already_exists

            CURR_HEADS_PATH="$ATTN_MAPS_FOLDER/av1_"$TMP_MODEL"_"$DS"_[$N_SHOTS]___forward_attn_maps_norms_mean.csv"
            echo $CURR_HEADS_PATH
            python perturbation_exp.py --model_name $MODEL --dataset_name $DS --out_folder $PERTURBATION_FOLDER --heads_list_path $CURR_HEADS_PATH --device "cuda:$DEVICE" --check_if_result_already_exists
        done
    done
done

