#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# general imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.pytorch_utils import Conv1D
import torch
import torch.nn as nn
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import random
import json

# another code we wrote
sys.path.append('../')
import llm_utils
import opt_utils
from plot_utils import plot_aux_wrapper
import exp_ra_utils


# In[ ]:


# try:
#     from function_vectors.src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
#     from function_vectors.src.utils.eval_utils import decode_to_vocab, sentence_eval, n_shot_eval_no_intervention, get_answer_id
# except Exception as error:
#     print('could not import from function_vectors package with the following error:')
#     print(error)
#     print('Make sure you first pull relevant submodules. See README.md for more info.')


# In[ ]:


START_TIME = time.strftime("%Y/%m/%d-%H:%M:%S")
# DEBUG_FLAG = True  # for local testing
DEBUG_FLAG = False  # automatically set below

try:
    DEBUG_FLAG = torch.backends.mps.is_available()  # since only Apple M1-3 supports this, we can use this as a flag for local testing
except:
    pass
if DEBUG_FLAG:
    print('*'*40)
    print(f'    DEBUG MODE [{START_TIME}]')
    print('*'*40)


# In[ ]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='gpt2' if DEBUG_FLAG else 'gpt2-xl')
parser.add_argument('--model_args', type=str, default='')
parser.add_argument('--out_folder', type=str, default='/tmp_plots1')
parser.add_argument('--postfix_name', type=str, default='')
parser.add_argument('--disable_pad_token', action='store_true')
parser.add_argument('--root_data_dir', type=str, default='../function_vectors/dataset_files')
parser.add_argument('--dataset_name', type=str, default='person-sport')
parser.add_argument('--heads_list_path', type=str, default='index')  # provide path to a file with a list of heads to use (only 'rand' and 'index' do not require an actual file)
parser.add_argument('--n_shots_icl', type=str, default='0,5' if DEBUG_FLAG else '0,1,5,10')
parser.add_argument('--n_samples', type=int, default=5 if DEBUG_FLAG else 25)
parser.add_argument('--percentage_step', type=float, default=0.333 if DEBUG_FLAG else 0.1)
parser.add_argument('--metric_to_use', type=str, default='f1_score')  # f1_score, exact_match_score, first_word_score. Not really used
parser.add_argument('--prefixes', help='Prompt template prefixes to be used', type=json.loads, required=False, default={"input":"Q:", "output":"A:", "instructions":""})
parser.add_argument('--separators', help='Prompt template separators to be used', type=json.loads, required=False, default={"input":"\n", "output":"\n\n", "instructions":""})    
parser.add_argument('--device', type=str, default=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--check_if_result_already_exists', action='store_true')


# In[ ]:


args, unknown = parser.parse_known_args()
print('unknown args:', unknown)
print('args:', args)


# In[ ]:


n_shots_icl = [int(x) for x in args.n_shots_icl.split(',')]
print(f'n_shots_icl: {n_shots_icl}')
metric_to_use = args.metric_to_use
n_samples = args.n_samples


# In[ ]:


# the prefix is a code for which version of the code we use in this notebook
output_prefix = f'per1_{args.model_name.replace("/", "-")}_{args.dataset_name}_[{args.n_shots_icl}]_{n_samples}_{args.postfix_name}_'

print(f'output_prefix: {output_prefix}')
plot_wrapper = plot_aux_wrapper(output_folder=args.out_folder, 
                            output_prefix=output_prefix,
                            local_show=True)


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if not args.disable_pad_token:
    print(f'adding pad token: {tokenizer.eos_token}')
    tokenizer.pad_token = tokenizer.eos_token
    
try:
    os.environ["TOKENIZERS_PARALLELISM"] = "true"  # not blocking, just to prevent warnings and faster tokenization
except:
    pass
padding_flag = tokenizer.pad_token_id is not None  # this is up to if the tokenizer was configured with padding or not


# In[ ]:


device = torch.device(args.device)
print(f'Using device: {device} [cuda available? => {torch.cuda.is_available()}, cuda version: {torch.version.cuda}, args.device = "{args.device}"]')


# In[ ]:


model_extra_args = {}
for arg in args.model_args.split(','):
    if arg == '':
        continue
    k, v = arg.split('=')
    model_extra_args[k] = v
print(f'model_extra_args: {model_extra_args}')

model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_extra_args).eval().requires_grad_(False).to(device)
model_aux = llm_utils.model_extra(model=model, device=device)
model_config = copy.deepcopy(model.config)
config = model_aux.config
# del model

n_embd = model_aux.n_embd
n_head = model_aux.n_head
head_size = model_aux.head_size
n_layer = model_aux.n_layer

pad_k = model_aux.pad_k
pad_v = model_aux.pad_v

# params_names_filter = opt_utils.get_filter_by_name(args.filter_layers)
params_names_filter = opt_utils.get_filter_by_name('attn_only')
# params_names_filter = lambda x: ('attn' in x or 'attention' in x) and 'weight' in x

show_every_n_layer = 1 if n_layer <= 12 else 2 if n_layer <= 24 else 4


# In[ ]:


print(model.config)


# In[ ]:


print(model)


# In[ ]:


is_llama = 'llama' in args.model_name or 'facebook/opt' in args.model_name  # llama and opt have similar configs and tokenization
prepend_bos = not is_llama

prefixes=args.prefixes
separators=args.separators

compute_ppl=False
shuffle_labels=False
generate_str=False


# In[ ]:


dataset = exp_ra_utils.data_loading_wrapper(args.dataset_name, 
                                            root_data_dir=args.root_data_dir,
                                            seed=args.seed)


# In[ ]:


sentence_example = exp_ra_utils.data_print_and_test_example(model, tokenizer=tokenizer,
                                                            dataset=dataset,
                                                            prepend_bos=prepend_bos,
                                                            prefixes=prefixes,
                                                            separators=separators,
                                                            shuffle_labels=shuffle_labels)


# In[ ]:


try:
    model = model.to('cpu')  # extra safety for the following experiments
    del model
except:
    pass


# # Alter the Model

# In[ ]:


def exp_perturbation_ablation(percentage_of_heads, sort_list_of_heads):
    # explicitly create and delte the model to make sure we used the right config
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_extra_args).eval().requires_grad_(False).to(device)
    heads_to_use = sort_list_of_heads[:int(len(sort_list_of_heads)*percentage_of_heads)]
    print(f'going to use {len(heads_to_use)} heads for {percentage_of_heads}')
    print(f'first 3 heads: {heads_to_use[:5]}')
    print(f'last 3 heads: {heads_to_use[-5:]}')
    used_heads = {f'{x[0]}_{x[1]}': False for x in heads_to_use}
    for layer_index in range(n_layer):
        Wo = llm_utils.rgetattr(model, config.attn_o.format(layer_index))
        for head_index in range(n_head):
            if f'{layer_index}_{head_index}' in used_heads:
                # print(f'{layer_index}_{head_index}')
                # if this head should be used - do not change it
                used_heads[f'{layer_index}_{head_index}'] = True  # just an indicator for debugging
            else: # if we do not want to use this head - zero it (equal to block its output)
                if type(Wo) == Conv1D:
                    Wo.weight[head_size*head_index:head_size*(head_index+1)] = 0
                elif type(Wo) == nn.Linear:
                    Wo.weight[:, head_size*head_index:head_size*(head_index+1)] = 0
                else:
                    raise ValueError(f'Unknown type of Wo: {type(Wo)} (currently only transformers.pytorch_utils.Conv1D and torch.nn.Linear are supported)')
    
    # res_icl = aux_eval_model(model, n_shots_list=n_shots_icl)
    res_icl = exp_ra_utils.aux_eval_model(model, tokenizer=tokenizer,
                                          model_name=args.model_name,
                                          dataset=dataset,
                                          metric_to_eval=metric_to_use,
                                          n_shots_list=n_shots_icl,
                                          prefixes=prefixes, separators=separators,
                                          compute_ppl=True,
                                          annotate=False)

    model = model.to('cpu')  # extra safety (not really needed)
    del model

    return res_icl


# In[ ]:


method_used = None
nshot_that_used_loaded_list = 777
if args.heads_list_path == 'rand':
    print(f'Going to randomize the heads list')
    method_used = 'rand'
    all_heads = [(layer_index, head_index, random.random()) for layer_index in range(n_layer) for head_index in range(n_head)]
elif args.heads_list_path == 'index':
    print(f'Going to use all heads by the order of their index (layer, head)')
    method_used = 'index'
    all_heads = [(layer_index, head_index, 0) for layer_index in range(n_layer) for head_index in range(n_head)]
elif 'attn_maps_norms_mean.csv' in args.heads_list_path:  # assuming backward or forward attention maps (mean across examples)
    print(f'\nGoing to load the heads list from {args.heads_list_path}')
    method_used = 'backward' if 'backward' in args.heads_list_path else 'forward'
    method_used = 'reverse' if 'reverse' in args.heads_list_path else method_used
    nshot_that_used_loaded_list = args.heads_list_path.split('[')[-1].split(']')[0]
    tmp = pd.read_csv(args.heads_list_path)
    all_heads = []
    for col in tmp.columns:
        if 'layer_' not in col:
            print(f'While processing the csv heads list file encounter the following column and skipped it: {col}')
            continue
        layer_index = int(col.split('_')[1])
        for row_index, row in tmp.iterrows():
            all_heads.append((layer_index, row_index, row[col]))
elif 'simple_casual_mediation' in args.heads_list_path:  # assuming simple casual mediation (our implementation)
    print(f'Loading heads list from {args.heads_list_path} (assuming simple casual mediation)')
    tmp = torch.load(args.heads_list_path)
    print(f'heads list info is in shape of {tmp.shape}')
    if len(tmp.shape) == 3:
        print(f'Assuming the [n_examples, n_layer, n_head] shape. averaging over n_examples')
        tmp = tmp.mean(dim=0)  # in case (n_examples, n_layer, n_head) that was used in previous versions (currenttly should be (n_layer, n_head))

    method_used = 'simple_causal_mediation'
    # now tmp should be of shape (n_layer, n_head). at every layer, head, we have the average indirect effect (AIE)
    print(f'Reduce heads list info to shape: {tmp.shape}. Assuming the [layer, head] entry holds the average indirect effect for that attention head')
    assert tmp.shape == (n_layer, n_head)
    nshot_that_used_loaded_list = args.heads_list_path.split('[')[-1].split(']')[0]
    all_heads = []
    for layer_index in range(n_layer):
        for head_index in range(n_head):
            all_heads.append((layer_index, head_index, tmp[layer_index, head_index].item()))
elif '_indirect_effect.pt' in args.heads_list_path:  # assuming Function Vectors implementation of indirect effect as tensor
    print(f'Loading heads list from {args.heads_list_path} (assuming indirect effect pt)')
    nshot_that_used_loaded_list = args.heads_list_path.split('LTO')[-1].split('AT')[-1].split('/')[0]
    tmp = torch.load(args.heads_list_path)
    print(f'heads list info is in shape of {tmp.shape}')
    if len(tmp.shape) == 4:  #  causal indirect effect (CIE) was taken for all tokens (if not - we assume it is for the last token only)
        tmp = tmp.mean(dim=-1)
        method_used = 'indirect_effect_AT'
    else:
        method_used = 'indirect_effect_LTO'

    # now tmp should be of shape (n_examples, n_layer, n_head). at every example, layer, head, we have the CIE
    tmp = tmp.mean(dim=0)
    # now tmp should be of shape (n_layer, n_head). at every layer, head, we have the average indirect effect (AIE)
    print(f'Reduce heads list info to shape: {tmp.shape}. Assuming the [layer, head] entry holds the average indirect effect for that attention head')
    assert tmp.shape == (n_layer, n_head)
    all_heads = []
    for layer_index in range(n_layer):
        for head_index in range(n_head):
            all_heads.append((layer_index, head_index, tmp[layer_index, head_index].item()))
else:
    raise ValueError(f'Unknown heads list format: {args.heads_list_path}')


print(f'Assuming the following method was used to create the heads list: {method_used}, {nshot_that_used_loaded_list}')
all_heads = sorted(all_heads, key=lambda x: x[2])
all_heads.reverse()  # sort in descending order (assuming the higher the value the more important the head is)
# please notice that after examining the orderer list, we also examin the reversed list (to make sure we did not miss anything)
# to verify the order (and what is it) - we save an extra file with the order of the heads (meta_data.json)


# In[ ]:


explicit_path_code = f'{output_prefix}_~ME:{method_used}~NS:{nshot_that_used_loaded_list}~'
df_path_out = os.path.join(args.out_folder, f'{explicit_path_code}_exp_perturbation_ablation.csv')
columns = ['percentage_of_heads']
for n_shots in n_shots_icl:
    columns.extend([f'{n_shots}_top_1', f'{n_shots}_top_2', f'{n_shots}_top_3'])
df = pd.DataFrame(data=[], columns=columns)
print(f'Going to save results to: {df_path_out}')
if args.check_if_result_already_exists and os.path.exists(df_path_out):
    print(f'File already exists: {df_path_out}. Terminating this run. If you wish to re-run, please remove the file or disable the flag --check_if_result_already_exists')
    print(f'Run args: {args}')
    print('Exit...')
    exit(0)
df.to_csv(df_path_out)  # to verify the file can be saved (currently empty)


# In[ ]:


range_of_percentages = np.arange(0, 1.01, args.percentage_step)


# In[ ]:


print(f'first 5 heads: {all_heads[:5]}')
print(f'last 5 heads: {all_heads[-5:]}')
meta_data = {'heads_order': all_heads, # the heads order that is saved in the meta data is before the reverse
             'heads_source': args.heads_list_path,
             'example_sentence': sentence_example,  # to verify we ran what we wanted
             'df_path_out': df_path_out,
             'range_of_percentages': [round(x, 2) for x in range_of_percentages],
             'run_args': str(args),
            'method_used': method_used,
            'nshot_that_used_loaded_list': nshot_that_used_loaded_list,
             'debug_flag_on': DEBUG_FLAG}
meta_data_out = os.path.join(args.out_folder, f'{explicit_path_code}_exp_perturbation_ablation_meta_data.json')
with open(meta_data_out, 'w') as f:
    json.dump(meta_data, f)


# In[ ]:


assert len(all_heads) == n_layer*n_head

all_results = {}
for percentage_of_heads in range_of_percentages:
    percentage_of_heads = round(percentage_of_heads, 2)  # to handle floating point errors
    res_icl = exp_perturbation_ablation(percentage_of_heads=percentage_of_heads, sort_list_of_heads=all_heads)
    
    for tmp_ in res_icl:
        del res_icl[tmp_]['clean_rank_list']  # examine the top-{1,2,3} accuracy (compute_top_k_accuracy)
        del res_icl[tmp_]['clean_ppl']
    all_results[percentage_of_heads] = res_icl


# In[ ]:


for percentage_of_heads, res in all_results.items():
    row = [percentage_of_heads]
    for n_shots in n_shots_icl:
        row.extend([res[n_shots]['clean_topk'][0][1], res[n_shots]['clean_topk'][1][1], res[n_shots]['clean_topk'][2][1]])
    df.loc[len(df)] = row
df.to_csv(df_path_out)


# In[ ]:


print('Finish with the experiment. Moving to apply the same process with the reversed heads list')
all_heads_rev = all_heads.copy()
all_heads_rev.reverse()
print(f'now the first 10 reversed heads: {all_heads_rev[:10]}')
print(f'now the last 10 reversed heads: {all_heads_rev[-10:]}')
assert len(all_heads_rev) == n_layer*n_head

all_results_rev = {}
for percentage_of_heads in range_of_percentages:
    percentage_of_heads = round(percentage_of_heads, 2)  # to handle floating point errors
    # print(f'percentage_of_heads: {percentage_of_heads}')
    res_icl = exp_perturbation_ablation(percentage_of_heads=percentage_of_heads, sort_list_of_heads=all_heads_rev)
    for tmp_ in res_icl:
        del res_icl[tmp_]['clean_rank_list']
        del res_icl[tmp_]['clean_ppl']
    all_results_rev[percentage_of_heads] = res_icl

df_path_out_rev = df_path_out.replace('.csv', '_REV.csv')
print(f'Going to save REVERSED ORDER results to: {df_path_out_rev}')
df_rev = pd.DataFrame(data=[], columns=columns)
for percentage_of_heads, res in all_results_rev.items():
    row = [percentage_of_heads]
    for n_shots in n_shots_icl:
        row.extend([res[n_shots]['clean_topk'][0][1], res[n_shots]['clean_topk'][1][1], res[n_shots]['clean_topk'][2][1]])
    df_rev.loc[len(df_rev)] = row
df_rev.to_csv(df_path_out_rev)  


# In[ ]:


print('DONE')


# In[ ]:




