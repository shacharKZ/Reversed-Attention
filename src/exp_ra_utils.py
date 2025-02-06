import os
import sys
import torch

sys.path.append('../')
try:
    from function_vectors.src.utils.prompt_utils import load_dataset, word_pairs_to_prompt_data, create_prompt
    from function_vectors.src.utils.eval_utils import decode_to_vocab, sentence_eval, n_shot_eval_no_intervention, get_answer_id
except Exception as error:
    print('could not import from function_vectors package with the following error:')
    print(error)
    print('Make sure you first pull relevant submodules. See README.md for more info.')


def get_forward_and_reversed_attn(collector, layer_index, head_index, config, head_size):
    attn_map = collector['attentions'][layer_index][0][head_index].clone().detach().cpu()  # forward pass attn map

    values = collector['kv_cache'][layer_index][1][0][head_index].clone().detach().cpu()

    vjp_output_backpropogated = collector['grad'][layer_index][config.attn_o]['input']
    vjp_output_backpropogated = torch.stack(vjp_output_backpropogated, dim=0)
    vjp_output_backpropogated = vjp_output_backpropogated[:, head_index*head_size:(head_index+1)*head_size].clone().detach().cpu()

    rev_logits = vjp_output_backpropogated @ values.T

    rev_attn = attn_map * (rev_logits.T - torch.diag(attn_map@rev_logits.T)).T / (head_size**0.5)

    return (attn_map, rev_attn)    


def data_loading_wrapper(dataset_name, seed, root_data_dir, print_examples=True):
    dataset = load_dataset(dataset_name, seed=seed,
                    root_data_dir=root_data_dir,  # from function_vectors
                    extra_data_folder=['abstractive_const_len', 'extractive_const_len'])  # added ds that are very much based on the original ds
    if print_examples:
        print(f'Load dataset: {dataset_name} with {len(dataset["train"])} train, {len(dataset["valid"])} valid and {len(dataset["test"])} test samples')
        print(f'Example: 1) {dataset["train"][1]}')
        print(f'Example: 2) {dataset["train"][2]}')
    
    return dataset


def data_print_and_test_example(model, tokenizer, dataset, prepend_bos, prefixes, separators, shuffle_labels):

    # Sample ICL example pairs, and a test word
    word_pairs = dataset['train'][:5]
    test_pair = dataset['test'][0]

    prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=prepend_bos,
                                            prefixes=prefixes, separators=separators)
    
    sentence = create_prompt(prompt_data)
    print("ICL prompt:\n", repr(sentence), '\n\n')

    # shuffled_prompt_data = word_pairs_to_prompt_data(word_pairs, query_target_pair=test_pair, prepend_bos_token=True, shuffle_labels=True)
    # shuffled_sentence = create_prompt(shuffled_prompt_data)
    # print("Shuffled ICL Prompt:\n", repr(shuffled_sentence), '\n\n')

    zeroshot_prompt_data = word_pairs_to_prompt_data({'input':[], 'output':[]}, query_target_pair=test_pair, 
                                                    prepend_bos_token=prepend_bos, shuffle_labels=shuffle_labels,
                                                    prefixes=prefixes, separators=separators)
    zeroshot_sentence = create_prompt(zeroshot_prompt_data)
    print("Zero-Shot Prompt:\n", repr(zeroshot_sentence))


    # Check model's ICL answer
    clean_logits = sentence_eval(sentence, [test_pair['output']], model, tokenizer, compute_nll=False)

    print("Input Sentence:", repr(sentence), '\n')
    print(f"Input Query: {repr(test_pair['input'])}, Target: {repr(test_pair['output'])}\n")
    print("ICL Prompt Top K Vocab Probs:\n", decode_to_vocab(clean_logits, tokenizer, k=5), '\n')

    # Intervention on the zero-shot prompt
    # clean_logits, interv_logits = function_vector_intervention(zeroshot_sentence, [test_pair['output']], EDIT_LAYER, FV, model, model_config, tokenizer)
    clean_logits = sentence_eval(zeroshot_sentence, [test_pair['output']], model, tokenizer, compute_nll=False)

    print("Input Sentence:", repr(zeroshot_sentence), '\n')
    print(f"Input Query: {repr(test_pair['input'])}, Target: {repr(test_pair['output'])}\n")
    print("Zero-Shot Prompt Top K Vocab Probs:\n", decode_to_vocab(clean_logits, tokenizer, k=5), '\n')

    # return an example sentence
    return sentence


def aux_eval_model(model, tokenizer, model_name, dataset, metric_to_eval, prefixes, separators, n_shots_list=[0, 1, 5, 10], compute_ppl=True, annotate=False):
    res_icl = {}
    for n_shots in n_shots_list:
        # fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config=model_config, tokenizer=tokenizer, compute_ppl=False, test_split='test')
        fs_results = n_shot_eval_no_intervention(dataset=dataset, n_shots=n_shots, model=model, model_config={'name_or_path':model_name}, 
                                                 tokenizer=tokenizer, compute_ppl=compute_ppl, test_split='test', metric=metric_to_eval,
                                                 prefixes=prefixes, separators=separators,
                                                 annotate=annotate)
        print(f"Few-Shot Results for {n_shots}:")
        # print(fs_results)
        print(fs_results['clean_topk'])
        res_icl[n_shots] = fs_results
    return res_icl
