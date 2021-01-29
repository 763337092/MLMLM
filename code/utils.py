import torch
import pickle
import numpy as np

def save_pickle(dic, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(dic, f)

def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        message_dict = pickle.load(f)
    return message_dict

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
