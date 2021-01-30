import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def load_data(root_path, debug):
    if debug:
        train = pd.read_csv(f'{root_path}/train.csv', nrows=500)
        valid = pd.read_csv(f'{root_path}/valid.csv', nrows=100)
        test = pd.read_csv(f'{root_path}/test.csv', nrows=100)
    else:
        train = pd.read_csv(f'{root_path}/train.csv')
        valid = pd.read_csv(f'{root_path}/valid.csv')
        test = pd.read_csv(f'{root_path}/test.csv')
    
    return train, valid, test

##### Prepare Dataset
def make_train_dataset(df, tokenizer, max_ent_length, max_seq_length):
    mask_idx = [tokenizer.convert_tokens_to_ids(tokenizer.mask_token) for _ in range(max_ent_length)]

    input_ids = []
    label_ids = []
    start_end_idxs = []
    for example in tqdm(df.values):
        entity1, desc1, entity2, desc2, relation = example

        entity1_idx = tokenizer.encode_plus(str(entity1), add_special_tokens=False)['input_ids']
        entity2_idx = tokenizer.encode_plus(str(entity2), add_special_tokens=False)['input_ids']
        desc1_idx = tokenizer.encode_plus(str(desc1), add_special_tokens=False)['input_ids']
        desc2_idx = tokenizer.encode_plus(str(desc2), add_special_tokens=False)['input_ids']
        relation_idx = tokenizer.encode_plus(str(relation), add_special_tokens=False)['input_ids']
        # print('entity1_idx: ', entity1_idx)
        # print('entity2_idx: ', entity2_idx)
        # print('relation_idx: ', relation_idx)

        ## left predict right
        inputs_ids_1 = [tokenizer.cls_token_id] + entity1_idx + desc1_idx + relation_idx + mask_idx + [tokenizer.sep_token_id]
        padding_length = max(0, max_seq_length - len(inputs_ids_1))
        inputs_ids_1 = inputs_ids_1 + ([tokenizer.pad_token_id] * padding_length)
        inputs_ids_1 = inputs_ids_1[:max_seq_length]

        entity2_target = entity2_idx + [tokenizer.pad_token_id] * (max_ent_length - len(entity2_idx))

        target_ids_1 = [tokenizer.pad_token_id] * (len(entity1_idx) + len(desc1_idx) + len(relation_idx) + 1) + entity2_target
        padding_length = max(0, max_seq_length - len(target_ids_1))
        target_ids_1 = target_ids_1 + ([tokenizer.pad_token_id] * padding_length)
        target_ids_1 = target_ids_1[:max_seq_length]
        # print('target_ids_1: ', target_ids_1)

        input_ids.append(inputs_ids_1)
        label_ids.append(target_ids_1)


        ## right predict left
        inputs_ids_2 = [tokenizer.cls_token_id] + mask_idx + relation_idx + entity2_idx + desc2_idx + [tokenizer.sep_token_id]
        padding_length = max(0, max_seq_length - len(inputs_ids_2))
        inputs_ids_2 = inputs_ids_2 + ([tokenizer.pad_token_id] * padding_length)
        inputs_ids_2 = inputs_ids_2[:max_seq_length]

        entity1_target = entity1_idx + [tokenizer.pad_token_id] * (max_ent_length - len(entity1_idx))

        target_ids_2 = [tokenizer.pad_token_id] + entity1_target
        padding_length = max(0, max_seq_length - len(target_ids_2))
        target_ids_2 = target_ids_2 + ([tokenizer.pad_token_id] * padding_length)
        target_ids_2 = target_ids_2[:max_seq_length]

        input_ids.append(inputs_ids_2)
        label_ids.append(target_ids_2)

    label_ids = torch.tensor(label_ids, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    tensor_dataset = TensorDataset(input_ids, label_ids)

    return tensor_dataset

def make_infer_dataset(df, tokenizer, max_ent_length, max_seq_length):
    mask_idx = [tokenizer.convert_tokens_to_ids(tokenizer.mask_token) for _ in range(max_ent_length)]

    input_ids = []
    label_ids = []
    start_end_idxs = []
    for example in tqdm(df.values):
        entity1, desc1, entity2, desc2, relation = example

        entity1_idx = tokenizer.encode_plus(str(entity1), add_special_tokens=False)['input_ids']
        entity2_idx = tokenizer.encode_plus(str(entity2), add_special_tokens=False)['input_ids']
        desc1_idx = tokenizer.encode_plus(str(desc1), add_special_tokens=False)['input_ids']
        desc2_idx = tokenizer.encode_plus(str(desc2), add_special_tokens=False)['input_ids']
        relation_idx = tokenizer.encode_plus(str(relation), add_special_tokens=False)['input_ids']
        # print('entity1_idx: ', entity1_idx)
        # print('entity2_idx: ', entity2_idx)
        # print('relation_idx: ', relation_idx)

        ## left predict right
        inputs_ids_1 = [tokenizer.cls_token_id] + entity1_idx + desc1_idx + relation_idx + mask_idx + [tokenizer.sep_token_id]
        padding_length = max(0, max_seq_length - len(inputs_ids_1))
        inputs_ids_1 = inputs_ids_1 + ([tokenizer.pad_token_id] * padding_length)
        inputs_ids_1 = inputs_ids_1[-max_seq_length:]
        # print('inputs_ids_1: ', inputs_ids_1)

        entity2_target = entity2_idx + [tokenizer.pad_token_id] * (max_ent_length - len(entity2_idx))
        # print('entity2_target: ', entity2_target)

        target_ids_1 = [tokenizer.pad_token_id] * (len(entity1_idx) + len(desc1_idx) + len(relation_idx) + 1) + entity2_target
        padding_length = max(0, max_seq_length - len(target_ids_1))
        overflow_length = max(0, len(target_ids_1) - max_seq_length)
        target_ids_1 = target_ids_1 + ([tokenizer.pad_token_id] * padding_length)
        target_ids_1 = target_ids_1[-max_seq_length:]
        # print('target_ids_1: ', target_ids_1)

        input_ids.append(inputs_ids_1)
        label_ids.append(target_ids_1)
        start_end_idxs.append([len(entity1_idx) + len(desc1_idx) + len(relation_idx) + 1 - overflow_length,
                               len(entity1_idx) + len(desc1_idx) + len(relation_idx) + 1 - overflow_length + len(mask_idx)])

        ## right predict left
        inputs_ids_2 = [tokenizer.cls_token_id] + mask_idx + relation_idx + entity2_idx + desc2_idx + [tokenizer.sep_token_id]
        padding_length = max(0, max_seq_length - len(inputs_ids_2))
        inputs_ids_2 = inputs_ids_2 + ([tokenizer.pad_token_id] * padding_length)
        inputs_ids_2 = inputs_ids_2[:max_seq_length]

        entity1_target = entity1_idx + [tokenizer.pad_token_id] * (max_ent_length - len(entity1_idx))

        target_ids_2 = [tokenizer.pad_token_id] + entity1_target
        padding_length = max(0, max_seq_length - len(target_ids_2))
        target_ids_2 = target_ids_2 + ([tokenizer.pad_token_id] * padding_length)
        target_ids_2 = target_ids_2[:max_seq_length]

        input_ids.append(inputs_ids_2)
        label_ids.append(target_ids_2)
        start_end_idxs.append([1, 1 + len(mask_idx)])
        # break
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    start_end_idxs = torch.tensor(start_end_idxs, dtype=torch.long)
    tensor_dataset = TensorDataset(input_ids, label_ids, start_end_idxs)

    return tensor_dataset
