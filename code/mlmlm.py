import os
import math
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel, get_linear_schedule_with_warmup, AdamW
from transformers.modeling_bert import BertLayerNorm

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

DATA_PATH = '../data/'
PRETRAINED_MODEL_PATH = '../pretrained/bert-base-uncased/'
OUTPUT_PATH = '../output/'
DEBUG = True
if DEBUG:
    OUTPUT_PATH += '/debug'
    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
        EPOCHS = 1

MAX_NAME_LENGTH = 24
MAX_SEQUENCE_LENGTH = 64
BATCH_SIZE = 64
WEIGHT_DECAY = 1e-5
WARMUP_RATIO = 0.
LEARNING_RATE = 1e-5
N_GPU = 1
EPOCHS = 30
GRAD_NORM = 1.0
EARLYSTOP_NUM = 3

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
    # with gzip.open(save_path, 'wb') as f:
        pickle.dump(dic, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
    # with gzip.open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict

tokenizer = BertTokenizer.from_pretrained(f'{PRETRAINED_MODEL_PATH}/vocab.txt')

##### Load Data
if DEBUG:
    train = pd.read_csv(f'{DATA_PATH}/train.csv', nrows=500)
    valid = pd.read_csv(f'{DATA_PATH}/valid.csv', nrows=100)
    test = pd.read_csv(f'{DATA_PATH}/test.csv', nrows=100)
else:
    train = pd.read_csv(f'{DATA_PATH}/train.csv')
    valid = pd.read_csv(f'{DATA_PATH}/valid.csv')
    test = pd.read_csv(f'{DATA_PATH}/test.csv')
print(f'Train|Valid|Test: {len(train)}|{len(valid)}|{len(test)}')

# All entities
all_entities = list(set(train.entity1.unique().tolist() + train.entity2.unique().tolist()))
print(f'All entities num: {len(all_entities)}')
id2entity = {idx: entity for idx, entity in enumerate(all_entities)}
entity2token_id = {entity: tokenizer.encode_plus(str(entity), add_special_tokens=False)['input_ids'] for entity in all_entities}

print(train.head())

##### Prepare Dataset
def make_train_dataset(df):
    mask_idx = [tokenizer.convert_tokens_to_ids(tokenizer.mask_token) for _ in range(MAX_NAME_LENGTH)]

    input_ids = []
    label_ids = []
    start_end_idxs = []
    for example in tqdm(df.values):
        entity1, entity2, relation = example

        entity1_idx = tokenizer.encode_plus(str(entity1), add_special_tokens=False)['input_ids']
        entity2_idx = tokenizer.encode_plus(str(entity2), add_special_tokens=False)['input_ids']
        relation_idx = tokenizer.encode_plus(str(relation), add_special_tokens=False)['input_ids']
        # print('entity1_idx: ', entity1_idx)
        # print('entity2_idx: ', entity2_idx)
        # print('relation_idx: ', relation_idx)

        ## left predict right
        inputs_ids_1 = [tokenizer.cls_token_id] + entity1_idx + relation_idx + mask_idx + [tokenizer.sep_token_id]
        padding_length = max(0, MAX_SEQUENCE_LENGTH - len(inputs_ids_1))
        inputs_ids_1 = inputs_ids_1 + ([tokenizer.pad_token_id] * padding_length)
        inputs_ids_1 = inputs_ids_1[:MAX_SEQUENCE_LENGTH]

        entity2_target = entity2_idx + [tokenizer.pad_token_id] * (MAX_NAME_LENGTH - len(entity2_idx))

        target_ids_1 = [tokenizer.pad_token_id] * (len(entity1_idx) + len(relation_idx) + 1) + entity2_target
        padding_length = max(0, MAX_SEQUENCE_LENGTH - len(target_ids_1))
        target_ids_1 = target_ids_1 + ([tokenizer.pad_token_id] * padding_length)
        target_ids_1 = target_ids_1[:MAX_SEQUENCE_LENGTH]
        # print('target_ids_1: ', target_ids_1)

        input_ids.append(inputs_ids_1)
        label_ids.append(target_ids_1)


        ## right predict left
        inputs_ids_2 = [tokenizer.cls_token_id] + mask_idx + relation_idx + entity2_idx + [tokenizer.sep_token_id]
        padding_length = max(0, MAX_SEQUENCE_LENGTH - len(inputs_ids_2))
        inputs_ids_2 = inputs_ids_2 + ([tokenizer.pad_token_id] * padding_length)
        inputs_ids_2 = inputs_ids_2[:MAX_SEQUENCE_LENGTH]

        entity1_target = entity1_idx + [tokenizer.pad_token_id] * (MAX_NAME_LENGTH - len(entity1_idx))

        target_ids_2 = [tokenizer.pad_token_id] + entity1_target
        padding_length = max(0, MAX_SEQUENCE_LENGTH - len(target_ids_2))
        target_ids_2 = target_ids_2 + ([tokenizer.pad_token_id] * padding_length)
        target_ids_2 = target_ids_2[:MAX_SEQUENCE_LENGTH]

        input_ids.append(inputs_ids_2)
        label_ids.append(target_ids_2)

    label_ids = torch.tensor(label_ids, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    tensor_dataset = TensorDataset(input_ids, label_ids)

    return tensor_dataset

def make_infer_dataset(df):
    mask_idx = [tokenizer.convert_tokens_to_ids(tokenizer.mask_token) for _ in range(MAX_NAME_LENGTH)]

    input_ids = []
    label_ids = []
    start_end_idxs = []
    for example in tqdm(df.values):
        entity1, entity2, relation = example

        entity1_idx = tokenizer.encode_plus(str(entity1), add_special_tokens=False)['input_ids']
        entity2_idx = tokenizer.encode_plus(str(entity2), add_special_tokens=False)['input_ids']
        relation_idx = tokenizer.encode_plus(str(relation), add_special_tokens=False)['input_ids']
        # print('entity1_idx: ', entity1_idx)
        # print('entity2_idx: ', entity2_idx)
        # print('relation_idx: ', relation_idx)

        ## left predict right
        inputs_ids_1 = [tokenizer.cls_token_id] + entity1_idx + relation_idx + mask_idx + [tokenizer.sep_token_id]
        padding_length = max(0, MAX_SEQUENCE_LENGTH - len(inputs_ids_1))
        inputs_ids_1 = inputs_ids_1 + ([tokenizer.pad_token_id] * padding_length)
        inputs_ids_1 = inputs_ids_1[-MAX_SEQUENCE_LENGTH:]
        # print('inputs_ids_1: ', inputs_ids_1)

        entity2_target = entity2_idx + [tokenizer.pad_token_id] * (MAX_NAME_LENGTH - len(entity2_idx))
        # print('entity2_target: ', entity2_target)

        target_ids_1 = [tokenizer.pad_token_id] * (len(entity1_idx) + len(relation_idx) + 1) + entity2_target
        padding_length = max(0, MAX_SEQUENCE_LENGTH - len(target_ids_1))
        overflow_length = max(0, len(target_ids_1) - MAX_SEQUENCE_LENGTH)
        target_ids_1 = target_ids_1 + ([tokenizer.pad_token_id] * padding_length)
        target_ids_1 = target_ids_1[-MAX_SEQUENCE_LENGTH:]
        # print('target_ids_1: ', target_ids_1)

        input_ids.append(inputs_ids_1)
        label_ids.append(target_ids_1)
        start_end_idxs.append([len(entity1_idx) + len(relation_idx) + 1 - overflow_length,
                               len(entity1_idx) + len(relation_idx) + 1 - overflow_length + len(mask_idx)])

        ## right predict left
        inputs_ids_2 = [tokenizer.cls_token_id] + mask_idx + relation_idx + entity2_idx + [tokenizer.sep_token_id]
        padding_length = max(0, MAX_SEQUENCE_LENGTH - len(inputs_ids_2))
        inputs_ids_2 = inputs_ids_2 + ([tokenizer.pad_token_id] * padding_length)
        inputs_ids_2 = inputs_ids_2[:MAX_SEQUENCE_LENGTH]

        entity1_target = entity1_idx + [tokenizer.pad_token_id] * (MAX_NAME_LENGTH - len(entity1_idx))

        target_ids_2 = [tokenizer.pad_token_id] + entity1_target
        padding_length = max(0, MAX_SEQUENCE_LENGTH - len(target_ids_2))
        target_ids_2 = target_ids_2 + ([tokenizer.pad_token_id] * padding_length)
        target_ids_2 = target_ids_2[:MAX_SEQUENCE_LENGTH]

        input_ids.append(inputs_ids_2)
        label_ids.append(target_ids_2)
        start_end_idxs.append([1, 1 + len(mask_idx)])
        # break
    label_ids = torch.tensor(label_ids, dtype=torch.long)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    start_end_idxs = torch.tensor(start_end_idxs, dtype=torch.long)
    tensor_dataset = TensorDataset(input_ids, label_ids, start_end_idxs)

    return tensor_dataset

train_dateset = make_train_dataset(df=train)
valid_dateset = make_infer_dataset(df=valid)
test_dateset = make_infer_dataset(df=test)
#
train_loader = DataLoader(train_dateset, shuffle=True, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_dateset, shuffle=False, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dateset, shuffle=False, batch_size=BATCH_SIZE)

##### Build Model
class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, epoch_score, model, model_path):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score: #  + self.delta
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # ema.apply_shadow()
            self.save_checkpoint(epoch_score, model, model_path)
            # ema.restore()
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            # print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            # if not DEBUG:
            torch.save(model.state_dict(), model_path)
        self.val_score = epoch_score

def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super(BertLMHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias

        return x

class BertMLMLM(BertPreTrainedModel):
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, path_dir, config):
        super().__init__(config)

        self.bert = BertModel.from_pretrained(path_dir, config=config) # , add_pooling_layer=False
        self.lm_head = BertLMHead(config)

        # self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        loss_mask=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        outputs = (prediction_scores,)

        if masked_lm_labels is not None:
            # Ignoring mask in backprop
            loss_fct = nn.CrossEntropyLoss(ignore_index=1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

device = torch.device('cuda:0')
config = BertConfig.from_pretrained(f'{PRETRAINED_MODEL_PATH}/config.json')
model = BertMLMLM(path_dir=PRETRAINED_MODEL_PATH, config=config)
model.to(device)

# Prepare optimizer
param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, eps=1e-8)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

mini_batch_size = min(1024//MAX_SEQUENCE_LENGTH, BATCH_SIZE)
gradient_accumulation_steps = math.ceil(BATCH_SIZE/mini_batch_size / N_GPU)
num_train_optimization_steps = int(math.ceil(len(train_dateset) / BATCH_SIZE / gradient_accumulation_steps) * EPOCHS)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=int(WARMUP_RATIO * num_train_optimization_steps),
                                            num_training_steps=num_train_optimization_steps)
# scheduler = None

##### Train&Eval
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
            topk_entity_idx = np.argsort(entity_prob_list)[::-1][:10]
            topk_entity = [id2entity[_id] for _id in topk_entity_idx]
            topk_entity_list.append(topk_entity)

    return np.array(topk_entity_list)

def eval_fn(target, pred_top10):
    index_list = []
    for label, pred in zip(target, pred_top10):
        if label in pred:
            this_idx = np.where(pred == label)[0][0]
        else:
            this_idx = 10
        index_list.append(this_idx)
    index_list = np.array(index_list)
    # print('index_list: ', index_list)
    hits1 = float(len(index_list[index_list < 1])) / len(index_list)
    hits3 = float(len(index_list[index_list < 3])) / len(index_list)
    hits10 = float(len(index_list[index_list < 10])) / len(index_list)
    return hits1, hits3, hits10

global_steps = 0
es = EarlyStopping(patience=EARLYSTOP_NUM, mode="max")
model_weights = f'{OUTPUT_PATH}/model.pth'
for ep in range(EPOCHS):
    train_fn(model, tokenizer, optimizer, scheduler, global_steps, train_loader)
    # print('global_steps: ', global_steps)
    # print('train_loss: ', train_loss)
    top10_val_pred = inference_fn(model, tokenizer, valid_loader)
    top10_test_pred = inference_fn(model, tokenizer, test_loader)

    valid_hits1, valid_hits3, valid_hits10 = eval_fn(valid[['entity1', 'entity2']].values.flatten(), top10_val_pred)
    test_hits1, test_hits3, test_hits10 = eval_fn(test[['entity1', 'entity2']].values.flatten(), top10_test_pred)
    print(f'Epoch{ep:3}: valid_hits@1={valid_hits1:.4f} valid_hits@3={valid_hits3:.4f} valid_hits@10={valid_hits10:.4f} '
          f'test_hits@1={test_hits1:.4f} test_hits@3={test_hits3:.4f} test_hits@10={test_hits10:.4f}')

    es(valid_hits1, model, model_path=model_weights)
    if es.early_stop:
        print("Early stopping")
        break

