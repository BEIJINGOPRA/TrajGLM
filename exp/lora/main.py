#!/usr/bin/env python
# coding=utf-8
"""
Training ChatGLMv2 based trajectory model by lora
"""

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import random
from torch import nn

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer_seq2seq import Seq2SeqTrainer
from arguments import ModelArguments, DataTrainingArguments

from peft import (
    LoraConfig, 
    PeftModel, 
    get_peft_model, 
    get_peft_model_state_dict, 
    set_peft_model_state_dict, 
    TaskType
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)
    
    # Load dataset
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file
        extension = data_args.test_file.split(".")[-1]

    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.road_source_path = model_args.road_source_path
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    with open(data_args.add_token_file_path,'r', encoding='UTF-8') as f:
        special_dict = json.load(f)
    tokenizer.add_tokens([key for key in special_dict.keys()])

##########################################################################################
    # 这里改造成 lora 的 checkpoint
    if model_args.ptuning_checkpoint is not None:
        # Loading LoRA checkpoint
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
##########################################################################################

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit)


    model = model.half()

###############################################################################################
    # 1. 多卡训练问题
    # 2. time 解耦，eta_time 变 second，最后的分类头变三个
    # 3. 搞清楚为什么 loss 一直都是 0
    # 4. 看一下 deepspeed、量化等方法加速大模型
###############################################################################################
    
    road_embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.load(model_args.road_source_path))).to(model.dtype).weight.data
    st_embedding = model.transformer.st_embedding.st_embedding.weight.data
    st_embedding[1:, :] = road_embedding
    model.transformer.st_embedding.st_embedding = nn.Embedding.from_pretrained(st_embedding)
    

    logger.info("*************************** load lora config **************************")
    target_modules_list = []
    for i in range(27):
        if i % 1 == 0:
            target_modules_list.append('{}.self_attention.query_key_value'.format(i))

    lora_config = LoraConfig(
        r = model_args.lora_r,
        lora_alpha = model_args.lora_alpha,
        # target_modules = [model_args.target_modules],
        target_modules = target_modules_list, 
        fan_in_fan_out=False,
        lora_dropout=0.05,
        inference_mode=False,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    for name, param in model.named_parameters():
        if "st_encoder._128_to_4096" in name:
            param.requires_grad = True
        elif "llm2st_encoder._4096_to_128" in name:
            param.requires_grad = True
        elif "time_embedding" in name:
            param.requires_grad = True
        elif "output_st_layer" in name:
            param.requires_grad = True
        elif "output_minute_time_layer" in name:
            param.requires_grad = True
        elif "modality_select_head" in name:
            param.requires_grad = True
        '''
        else:
            param.requires_grad = False
        '''

    model.print_trainable_parameters()
    for name, param in model.named_parameters():
        print(name, " : ", param.requires_grad)
    
    '''
    import pdb
    pdb.set_trace()
    '''

    # 模型生成的最大序列长度
    max_target_length = data_args.max_target_length

    prompt_column = ['path', 'traj_minute_indexs', 'traj_week_indexs']
    response_column = ['path', 'traj_minute_indexs', 'traj_week_indexs', 'traj_eta', 'traj_eta_list']

    def get_noisy_mask(traj_len, ratio = 0.3):
        '''
        return 返回一个traj_len长度的mask数组
        '''
        if ratio == 0.0:
            return (np.zeros(traj_len, dtype=int)).tolist()
        else:
            mask_ratio = random.uniform(0.1, ratio)
            mask = np.random.choice(np.array([0, 1]), size=traj_len, replace=True, p=(1 - mask_ratio, mask_ratio))
            ##### 保证每个序列的出发点不会被 mask 掉
            mask[0] = 0
            return mask.tolist()

    def get_mask_input_query(path_raw, time_minute_raw, time_week_raw, signal_ids):
        '''
        return query, answer
        query :[path_ids, time_minute_ids, time_week_ids]
        answer : 一个数组，包含 path_ids 中被 mask 掉的部分
        '''
        noise_mask = get_noisy_mask(len(path_raw), 0.3)

        query = []
        answer = []
        path_ids_query = []
        time_minute_ids_query = []
        time_week_ids_query = []

        path_ids_answer = []
        time_minute_ids_answer = []
        time_week_ids_answer = []

        for i in range(len(noise_mask)):
            if noise_mask[i] == 0:
                path_ids_query.append(path_raw[i])
                time_minute_ids_query.append(time_minute_raw[i])
                time_week_ids_query.append(time_week_raw[i])
            else:
                if noise_mask[i-1] == 0:
                    path_ids_query.append(signal_ids[0])
                    time_minute_ids_query.append(signal_ids[0])
                    time_week_ids_query.append(signal_ids[0])
                    # answer.append(signal_ids[1])
                    path_ids_answer.append(signal_ids[1])
                    time_minute_ids_answer.append(signal_ids[1])
                    time_week_ids_answer.append(signal_ids[1])
                # answer.append(path_raw[i])
                path_ids_answer.append(path_raw[i])
                time_minute_ids_answer.append(time_minute_raw[i])
                time_week_ids_answer.append(time_week_raw[i])

        query.append(path_ids_query + [signal_ids[1]])
        query.append(time_minute_ids_query + [signal_ids[1]])
        query.append(time_week_ids_query + [signal_ids[1]])

        # answer = answer if len(answer) == 0 else answer[1:]

        path_ids_answer = path_ids_answer if len(path_ids_answer) == 0 else path_ids_answer[1:]
        time_minute_ids_answer = time_minute_ids_answer if len(time_minute_ids_answer) == 0 else time_minute_ids_answer[1:]
        time_week_ids_answer = time_week_ids_answer if len(time_week_ids_answer) == 0 else time_week_ids_answer[1:]

        answer.append(path_ids_answer)
        answer.append(time_minute_ids_answer)
        answer.append(time_week_ids_answer)

        text_mask = []
        for i in range(len(query[0])):
            if query[0][i] < 64794:
                text_mask.append(1)
            else:
                text_mask.append(0)
        
        for i in range(len(answer[0])):
            if answer[0][i] < 64794:
                text_mask.append(1)
            else:
                text_mask.append(0)

        return query, answer, text_mask   

    def get_position_ids(input_path_ids, input_len, query_len, mask_ids, start_ids, answer):
        '''
        input_len : total length of the input sequence
        query_len : length of the query sequence which does not contain the last sop
        '''
        seq_len = len(input_path_ids)
        position_1 = []
        position_2 = []
        flag_index = []

        for i in range(seq_len):
            if i < query_len:
                position_1.append(i+1)
                if input_path_ids[i] == mask_ids and i > 2:
                    flag_index.append(i+1)
            elif i < input_len and i >= query_len:
                position_1.append(0)
            elif i >= input_len:
                position_1.append(query_len + 5)
            position_2.append(0)

        flag_location = 0
        if len(flag_index) == 0:
            position_1[input_len-2] = input_len - 1
            position_1[input_len-1] = input_len
            position_2[input_len-2] = 1
            position_2[input_len-1] = 2

            return position_1, position_2

        for i in range(query_len, input_len):
            if input_path_ids[i] == start_ids and i > 2:
                position_1[i] = flag_index[flag_location]
                position_2[i] = 1
                flag_location += 1

        flag_value = 0
        for i in range(query_len, input_len):
            if position_1[i] > 0:
                flag_value = position_1[i]
            i += 1
            while(position_1[i] == 0):
                position_1[i] = flag_value
                position_2[i] = position_2[i-1] + 1

        position_1[input_len-1] = 11
        position_2[input_len-1] = 1

        return position_1, position_2

    def preprocess_function_train(examples):
        '''
        这个函数面向的是 pretrain 阶段，pretrain是自监督训练，不需要 instruction 和 answer
        instruction 通过 build_prompt 构建， answer 从数据本身获得
        return: 
        input_ids : 
            text_ids:   [[gMASK], sop, prefix, pad_tokens, gMASK, pad_tokens, sop]   + [labels, eos] + [pad_ids] * pad_len
            path_ids:   [0, 0, [0] * prefix_len, path_tokens, 0, path_tokens, 0]   + [labels, 0] + [0] * pad_len
            minute_ids: [0, 0, [0] * prefix_len, minute_tokens, 0, week_tokens, 0] + [labels, 0] + [0] * pad_len
            week_ids:   [0, 0, [0] * prefix_len, week_tokens, 0, week_tokens, 0]   + [labels, 0] + [0] * pad_len
        
        attention_mask :
            attention_mask: [1] * (prefix_len  + path_ids_len) + [0] * (labels_len + pad_len)
            text_mask:      [1] * (prefix_len) + [1,0,...,1,0] + [1,0,..,0,1] + [1] * pad_len
            st_mask:        [0] * (prefix_len) + [0,1,...,0,1] + [0,1,..,1,0] + [0] * pad_len

        labels : 
            text_labels:   [-100] * prefix_len + [-100,   sop, -100,   -100,   sop, ..., eos] + [-100] * pad_len
            st_labels:     [-100] * prefix_len + [token1, -100,token2, token3, -100, .. -100] + [-100] * pad_len
            minute_labels: [-100] * prefix_len + [token1, -100,token2, token3, -100, .. -100] + [-100] * pad_len
        
        position_ids:
            position_1: [0,] * len_prefix + [index] + [input_len, ] * pad_len
            position_2: [0]  * len_prefix + [0,] + [input_len, ] * pad_len
        '''
        max_seq_length = data_args.max_source_length + data_args.max_target_length + 1

        model_inputs = {
            "input_ids" : [],
            "attention_mask": [],
            "labels": [],
            "position_ids": []
        }

        for i in range(len(examples[prompt_column[0]])):
            path_raw = examples[prompt_column[0]][i]
            time_minute_raw = examples[prompt_column[1]][i]
            time_week_raw = examples[prompt_column[2]][i]

            traj_eta_raw = examples[response_column[3]][i]
            traj_eta_list_raw = examples[response_column[4]][i]
            prefix_raw = ['[gMASK]', 'sop']
    
            signal_ids = tokenizer.convert_tokens_to_ids(['[gMASK]', 'sop'])
            prefix_ids = tokenizer.convert_tokens_to_ids(prefix_raw)
            path_ids = tokenizer.convert_tokens_to_ids(path_raw)[:data_args.max_source_length]
            time_minute_ids = tokenizer.convert_tokens_to_ids(time_minute_raw)[:data_args.max_source_length]
            time_week_ids = tokenizer.convert_tokens_to_ids(time_week_raw)[:data_args.max_source_length]

            query, answer, text_mask = get_mask_input_query(path_ids, time_minute_ids, time_week_ids, signal_ids)

            content_length = len(prefix_ids + query[0])
            answer_length = len(answer[0])
            input_path_ids = prefix_ids + query[0] + answer[0] + [tokenizer.eos_token_id]
            input_time_minute_ids = prefix_ids + query[1] + answer[1] + [tokenizer.eos_token_id]
            input_time_week_ids = prefix_ids + query[2] + answer[2] + [tokenizer.eos_token_id]
            
            path_labels = [tokenizer.pad_token_id] * content_length + answer[0] + [tokenizer.eos_token_id]
            time_minute_labels = [tokenizer.pad_token_id] * content_length + answer[1] + [tokenizer.eos_token_id]

            attention_mask = [0] * content_length

            pad_len = max_seq_length - len(input_path_ids)
            input_path_ids = input_path_ids + [tokenizer.pad_token_id] * pad_len
            input_time_minute_ids = input_time_minute_ids + [tokenizer.pad_token_id] * pad_len
            input_time_week_ids = input_time_week_ids + [tokenizer.pad_token_id] * pad_len
            
            attention_mask = attention_mask + [1] * (max_seq_length - content_length)
            text_mask = [1] * len(prefix_ids) + text_mask + [1] * (max_seq_length - content_length - answer_length)
            st_mask = [0 if text_mask[k] else 1 for k in range(len(text_mask))]

            path_labels = path_labels + [tokenizer.pad_token_id] * pad_len
            time_minute_labels = time_minute_labels + [tokenizer.pad_token_id] * pad_len

            st_labels, minute_labels, text_labels = [], [], []
            for j in range(len(path_labels)):
                if text_mask[j] == 1 and attention_mask[j] == 1:
                    text_labels.append(path_labels[j])
                else:
                    text_labels.append(tokenizer.pad_token_id)

            for j in range(len(path_labels)):
                if st_mask[j] == 1 and attention_mask[j] == 1:
                    st_labels.append(path_labels[j] - 64794)
                else:
                    st_labels.append(tokenizer.pad_token_id)
            
            ####### 先只写一个分钟级别的时间回归 #######
            for j in range(len(time_minute_labels)):
                if st_mask[j] == 1 and attention_mask[j] == 1:
                    minute_labels.append(time_minute_labels[j] - 105101)
                else:
                    minute_labels.append(tokenizer.pad_token_id)

            pure_text_ids = []
            pure_path_ids = []
            pure_minute_ids = []
            pure_week_ids = []

            for j in range(len(input_path_ids)):
                if st_mask[j] == 1:
                    pure_path_ids.append(input_path_ids[j] - 64794)
                else:
                    pure_path_ids.append(0)

            for j in range(len(input_path_ids)):
                if text_mask[j] == 1:
                    pure_text_ids.append(input_path_ids[j])
                else:
                    pure_text_ids.append(0)

            for j in range(len(input_path_ids)):
                if st_mask[j] == 1:
                    pure_minute_ids.append(input_time_minute_ids[j] - 105101)
                    pure_week_ids.append(input_time_week_ids[j] - 105101)
                else:
                    pure_minute_ids.append(0)
                    pure_week_ids.append(0)

            if data_args.ignore_pad_token_for_loss:
                text_labels = [(l if l != tokenizer.pad_token_id else -100) for l in text_labels]
                st_labels = [(l if l != tokenizer.pad_token_id else -100) for l in st_labels]
                minute_labels = [(l if l != tokenizer.pad_token_id else -100) for l in minute_labels]

            input_len = content_length + answer_length + 1
            query_len = content_length - 1
            mask_ids = tokenizer.convert_tokens_to_ids('[gMASK]')
            start_ids = tokenizer.convert_tokens_to_ids('sop')

            position_1, position_2 = get_position_ids(input_path_ids, input_len, query_len, mask_ids, start_ids, answer[0])

            model_inputs['input_ids'].append(np.vstack([pure_text_ids, pure_path_ids, pure_minute_ids, pure_week_ids]))
            model_inputs['attention_mask'].append(np.vstack([attention_mask, text_mask, st_mask]))
            model_inputs["labels"].append(np.vstack([text_labels, st_labels, minute_labels]))
            # model_inputs["position_ids"].append(np.vstack([position_1, position_2]))
            model_inputs["position_ids"].append(position_1)

        return model_inputs


    def preprocess_function_eval(examples):
        inputs, targets = [], []
        for i in range(len(examples[prompt_column])):
            if examples[prompt_column][i] and examples[response_column][i]:
                query = examples[prompt_column][i]
                history = examples[history_column][i] if history_column is not None else None
                prompt = tokenizer.build_prompt(query, history)
                inputs.append(prompt)
                targets.append(examples[response_column][i])

        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def print_dataset_example(example):
        for key in example.keys():
            print("{}_{}_ids".format(key, 1), " : ", example[key][0])
            print('{}_{}_token'.format(key, 1), " : ", tokenizer.decode(example[key][0]))
            print("seq_1 length", " : ", len(example[key][0]))

            print("{}_{}_ids".format(key, 2), " : ", example[key][1])
            print('{}_{}_token'.format(key, 2), " : ", tokenizer.decode(example[key][1]))
            print("seq_2 length", " : ", len(example[key][1]))

            print("{}_{}_ids".format(key, 3), " : ", example[key][2])
            print('{}_{}_token'.format(key, 3), " : ", tokenizer.decode(example[key][2]))
            print("seq_3 length", " : ", len(example[key][2]))

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function_train,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                # remove_columns = None,
                remove_columns = ['traj_minute_indexs', 'traj_eta', 'path', 'traj_eta_list', 'traj_week_indexs'],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
        # print_dataset_example(train_dataset[0])
    
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        print_dataset_example(eval_dataset[0])

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        print_dataset_example(predict_dataset[0])

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

#################################################################################
    # Metric
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        for pred, label in zip(decoded_preds, decoded_labels):
            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis) , ' '.join(reference))
            result = scores[0]
            
            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        return score_dict
#################################################################################

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_changed=model_args.pre_seq_len is not None
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=max_seq_length, temperature=0.95)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length, do_sample=True, top_p=0.7, temperature=0.95)
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                labels = [label.strip() for label in labels]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                        writer.write(f"{res}\n")
    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
