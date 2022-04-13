import os
import math
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel, get_linear_schedule_with_warmup, AdamW

from model import BertMLMLM
from processing import load_data, make_train_dataset, make_infer_dataset
from utils import EarlyStopping, eval_fn

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DATA_PATH = '../data/'
PRETRAINED_MODEL_PATH = '../pretrained/bert-base-uncased/'
OUTPUT_PATH = '../output/'

DEBUG = False
if DEBUG:
    OUTPUT_PATH += '/debug'
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
        EPOCHS = 1

MAX_ENTITY_LENGTH = 24
MAX_SEQUENCE_LENGTH = 64
BATCH_SIZE = 32
WEIGHT_DECAY = 0.1
WARMUP_RATIO = 0.
LEARNING_RATE = 2e-5
N_GPU = 1
EPOCHS = 25
GRAD_NORM = 1.0
EARLYSTOP_NUM = 5

def trim_tensor(input_ids, label_ids, tokenizer):
    max_len = torch.max(torch.sum((input_ids != tokenizer.pad_token_id), -1))
    input_ids = input_ids[:, :max_len]
    label_ids = label_ids[:, :max_len]
    return input_ids, label_ids

def train_fn(model, tokenizer, optimizer, scheduler, global_steps, train_loader):
    model.train()
    train_loss = []
    for (input_ids, label_ids) in tqdm(train_loader, desc='Train'):
    # for (input_ids, label_ids) in train_loader:
        input_ids, label_ids = trim_tensor(input_ids, label_ids, tokenizer)
        input_ids, label_ids = input_ids.to(device), label_ids.to(device)

        loss, _ = model(input_ids=input_ids, attention_mask=(input_ids != tokenizer.pad_token_id),
                     masked_lm_labels=label_ids)
        # print(_.shape)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()

        if (global_steps + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()

        train_loss.append(loss.mean().item())
        global_steps += 1
    return global_steps, train_loss

def inference_fn(model, tokenizer, data_loader):
    model.eval()
    topk_entity_list = []
    for (input_ids, label_ids, start_end_idxs) in tqdm(data_loader, desc='Infer'):
    # for (input_ids, label_ids, start_end_idxs) in data_loader:
        input_ids, label_ids = trim_tensor(input_ids, label_ids, tokenizer)
        input_ids, label_ids = input_ids.to(device), label_ids.to(device)

        _, output_prob = model(input_ids=input_ids, attention_mask=(input_ids != tokenizer.pad_token_id),
                     masked_lm_labels=label_ids)
        output_prob = output_prob.detach().cpu().numpy()
        label_ids = label_ids.detach().cpu().numpy()
        start_end_idxs = start_end_idxs.cpu().numpy()
        # print('input_ids: ', input_ids.shape)
        for j in range(len(input_ids)):
            # print('output_prob: ', output_prob.shape)
            preds = output_prob[j][start_end_idxs[j][0]: start_end_idxs[j][1]]
            target = label_ids[j][start_end_idxs[j][0]: start_end_idxs[j][1]]
            # target = np.swapaxes(target, 0, 1)
            # preds = np.argmax(preds, axis=1)
            # print('preds: ', preds)
            # print('target: ', target)
            entity_prob_list = []
            for _entity_id in range(len(id2entity)):
                _entity_token_id = entity2token_id[id2entity[_entity_id]]
                _entity_prob = preds[np.arange(len(_entity_token_id)), _entity_token_id]
                # print('_entity_prob: ', _entity_prob)
                entity_prob_list.append(np.mean(_entity_prob))
            topk_entity_idx = np.argsort(entity_prob_list)[::-1]#[:10]
            topk_entity = [id2entity[_id] for _id in topk_entity_idx]
            topk_entity_list.append(topk_entity)

    return np.array(topk_entity_list)

if __name__ == '__main__':
    ## Load Data
    train, valid, test = load_data(root_path=DATA_PATH, debug=DEBUG)
    print(f'Train|Valid|Test: {len(train)}|{len(valid)}|{len(test)}')

    ## Load Tokenizer
    tokenizer = BertTokenizer.from_pretrained(f'{PRETRAINED_MODEL_PATH}/vocab.txt')

    ## Process entities
    all_entities = list(set(
        train.entity1.unique().tolist() + train.entity2.unique().tolist() + 
        valid.entity1.unique().tolist() + valid.entity2.unique().tolist() + 
        test.entity1.unique().tolist() + test.entity2.unique().tolist()
        ))
    print(f'All entities num: {len(all_entities)}')
    id2entity = {idx: entity for idx, entity in enumerate(all_entities)}
    entity2token_id = {entity: tokenizer.encode_plus(str(entity), add_special_tokens=False)['input_ids'] for entity in
                       tqdm(all_entities, desc='Making entity tokens')}

    ## Make Datasets
    train_dateset = make_train_dataset(
        df=train,
        tokenizer=tokenizer,
        max_ent_length=MAX_ENTITY_LENGTH,
        max_seq_length=MAX_SEQUENCE_LENGTH,
    )
    valid_dateset = make_infer_dataset(
        df=valid,
        tokenizer=tokenizer,
        max_ent_length=MAX_ENTITY_LENGTH,
        max_seq_length=MAX_SEQUENCE_LENGTH,
    )
    test_dateset = make_infer_dataset(
        df=test,
        tokenizer=tokenizer,
        max_ent_length=MAX_ENTITY_LENGTH,
        max_seq_length=MAX_SEQUENCE_LENGTH,
    )
    train_loader = DataLoader(train_dateset, shuffle=True, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_dateset, shuffle=False, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dateset, shuffle=False, batch_size=BATCH_SIZE)

    ## Build Model
    device = torch.device('cuda:0')
    config = BertConfig.from_pretrained(f'{PRETRAINED_MODEL_PATH}/config.json')
    model = BertMLMLM(path_dir=PRETRAINED_MODEL_PATH, config=config)
    model.to(device)

    ## Prepare params
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': WEIGHT_DECAY},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    mini_batch_size = min(1024 // MAX_SEQUENCE_LENGTH, BATCH_SIZE)
    gradient_accumulation_steps = math.ceil(BATCH_SIZE / mini_batch_size / N_GPU)
    num_train_optimization_steps = int(
        math.ceil(len(train_dateset) / BATCH_SIZE / gradient_accumulation_steps) * EPOCHS)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(WARMUP_RATIO * num_train_optimization_steps),
                                                num_training_steps=num_train_optimization_steps)

    ## Train&Eval
    global_steps = 0
    es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
    model_weights = f'{OUTPUT_PATH}/model.pth'
    start_time = time.time()
    for ep in range(EPOCHS):
        train_fn(model, tokenizer, optimizer, scheduler, global_steps, train_loader)
        # print('global_steps: ', global_steps)
        # print('train_loss: ', train_loss)
        top10_val_pred = inference_fn(model, tokenizer, valid_loader)
        top10_test_pred = inference_fn(model, tokenizer, test_loader)

        valid_hits1, valid_hits3, valid_hits10, valid_MR, valid_MRR = eval_fn(valid[['entity1', 'entity2']].values.flatten(), top10_val_pred)
        test_hits1, test_hits3, test_hits10, test_MR, test_MRR = eval_fn(test[['entity1', 'entity2']].values.flatten(), top10_test_pred)
        print(
                f'Epoch{ep:3}: valid_hits@1={valid_hits1:.4f} valid_hits@3={valid_hits3:.4f} valid_hits@10={valid_hits10:.4f} valid_MR={valid_MR:.4f} valid_MRR={valid_MRR:.4f} '
                f'test_hits@1={test_hits1:.4f} test_hits@3={test_hits3:.4f} test_hits@10={test_hits10:.4f} test_MR={test_MR:.4f} test_MRR={test_MRR:.4f} '
            f'Time={(time.time()-start_time) / 60:.2f}min')

        es(valid_hits10, model, model_path=model_weights)
        if es.early_stop:
            print("Early stopping")
            break
