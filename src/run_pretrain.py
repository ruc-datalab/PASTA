import os
import sys
from tkinter.messagebox import NO 
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import torch.optim as optim
from torch.autograd import Variable
import sklearn
import torch
import torch.nn.functional as F
import math
import random
import re
import datetime
from transformers.hf_argparser import HfArgumentParser
from utils.args import ModelArguments, DataArguments
from utils.dataset import DataManager
from utils.pasta_mlm_model import *

class PASTA_Trainer:
    def __init__(self, model_args, data_args, device):
        if device is None:
            self.device = 'cuda'
        else:
            self.device = device
        self.model_args = model_args
        self.data_args = data_args
        if self.model_args.load_checkpoints != None:
            self.model = torch.load(self.model_args.load_checkpoints)
        else:
            self.model = ModelDefine(model_args)
        self.model.to(self.device)
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.data_args.tokenizer_path)
        self.train_loss = AverageMeter()
        self.updates = 0
        self.optimizer = optim.Adamax([p for p in self.model.parameters() if p.requires_grad], lr=model_args.learning_rate)
        self.dt = DataManager(data_args)
    def train(self):
        for i in range(self.model_args.epochs):
            print("=" * 30)
            print("epoch%d" % i)
            with tqdm(enumerate(self.dt.iter_batches(which="train", batch_size=self.model_args.train_batch_size)), ncols=80) as t:
                for batch_id, batch in t:
                    self.model.train()
                    input_ids, token_type_ids, attention_mask, labels = [
                        Variable(e).long().to(self.device) for e
                        in batch]
                    labels = torch.tensor(labels).unsqueeze(0)
                    logits,loss = self.model(
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        labels=labels) 
                    self.train_loss.update(loss.item(), self.model_args.train_batch_size)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    t.set_postfix(loss=self.train_loss.avg)
                    t.update(1)
                    self.updates += 1

            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset()

            if self.model_args.do_eval:
                self.validate(epoch=i, which='dev')

            if self.model_args.save_model:
                save_plm_path = os.path.join(self.model_args.save_model_path, f"pretrain_TB{self.model_args.train_batch_size}_LR{self.model_args.learning_rate}")
                if not os.path.exists(save_plm_path):
                    os.mkdir(save_plm_path)
                save_plm_path = os.path.join(save_plm_path, f"Epoch_{i+1}")
                if not os.path.exists(save_plm_path):
                    os.mkdir(save_plm_path)
                self.model.deberta.save_pretrained(save_plm_path)
                print(f"save epoch {i+1} to {save_plm_path}")


    def validate(self, which="dev", epoch=-1):
        self.test_loss = AverageMeter()
        def simple_accuracy(preds, labels):
            total = 0
            success = 0
            for pred, label in zip(preds, labels):
                for p, l in zip(pred, label):
                    if l == -100: #-100 for ignore token
                        continue
                    elif p == l:
                        success += 1
                        total += 1
                    else:
                        total += 1
            return success/total

        preds_label = []
        gold_label = []
        preds_examples = []
        gold_examples = []
        with tqdm(enumerate(self.dt.iter_batches(which=which, batch_size=self.model_args.eval_batch_size)), ncols=80) as t:
            for batch_id, batch in t:
                self.model.eval()
                input_ids, token_type_ids, attention_mask, labels = [
                        Variable(e).long().to(self.device) for e
                        in batch]

                labels = torch.tensor(labels).unsqueeze(0)
                logits,loss = self.model(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels)
                self.test_loss.update(loss.item(), self.model_args.eval_batch_size)
                preds = np.argmax(logits.detach().cpu().numpy(), axis=-1)
                if batch_id % 100 == 0:
                    pred_ids = list(preds[0])
                    gold_ids = batch[-1][0].numpy().tolist()
                    mask_pred_ids = []
                    mask_gold_ids = []
                    for index, id in enumerate(gold_ids):
                        if id != -100:
                            mask_pred_ids.append(pred_ids[index])
                            mask_gold_ids.append(id)
                    Pred_string = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(mask_pred_ids))
                    Gold_string = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(mask_gold_ids))
                    print(f"pred: {Pred_string}\ngold: {Gold_string}")
                    preds_examples.append(Pred_string)
                    gold_examples.append(Gold_string)
                preds_label.extend(preds)
                gold_label.extend(batch[-1])

        acc = simple_accuracy(preds_label, gold_label)
        print(f"epoch {epoch+1} test acc={acc} test loss={self.test_loss.avg}")
        with open(os.path.join(self.model_args.save_results_path, f"pretrain_TB{self.model_args.train_batch_size}_LR{self.model_args.learning_rate}.jsonl"), "a") as f:
            f.write(f"Epoch {epoch+1} test acc={acc} test loss={self.test_loss.avg}\n")
            f.write("Examples:\n")
            for pred_str, gold_str in zip(preds_examples, gold_examples):
                f.write(f"Pred: {pred_str}   Gold: {gold_str}\n")
        self.test_loss.reset()

class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataArguments)
    )
    model_args: ModelArguments
    data_args: DataArguments
    model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    setup_seed(0)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'gpu'
    Trainer = PASTA_Trainer(
        model_args=model_args, 
        data_args=data_args, 
        device=device
    )
    Trainer.train()
