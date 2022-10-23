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
from transformers.hf_argparser import HfArgumentParser
from utils.args import ModelArguments, DataArguments
from utils.dataset import DataManager
from utils.TabFV_model import ModelDefine

class TabFV_Trainer:
    def __init__(self, model_args, data_args, device):
        if device is None:
            self.device = 'cuda'
        else:
            self.device = device
        self.model_args = model_args
        self.data_args = data_args
        self.model = ModelDefine(model_args)
        self.model.to(self.device)
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

                    if batch_id == 10000:
                        print(f"middle: epoch {i}, train loss={self.train_loss.avg}")
                        self.validate(epoch=i, which='dev')

            print("epoch {}, train loss={}".format(i, self.train_loss.avg))
            self.train_loss.reset()
            if self.model_args.do_eval:
                self.validate(epoch=i, which='dev',save=True)
            if self.model_args.do_test:
                if self.data_args.dataset_name == 'tabfact':
                    self.validate(epoch=i, which='test',save=True)
                    self.validate(epoch=i, which='simple',save=True)
                    self.validate(epoch=i, which='complex',save=True)
                elif self.data_args.dataset_name == 'semtabfacts':
                    self.validate(epoch=i, which='test',save=True)
            if self.model_args.save_model:
                save_path = os.path.join(self.model_args.save_model_path, 'checkpoint_epoch_{}.pkl'.format(i+1))
                print(f"save epoch {i+1} to {save_path}")
                torch.save(self.model, save_path)            

    def validate(self, which="dev", epoch=-1, save=False):
        def simple_accuracy(preds1, labels1):
            correct_num = sum([1.0 if p1 == p2 else 0.0 for p1, p2 in zip(preds1, labels1)])
            return correct_num / len(preds1)

        preds_label = []
        gold_label = []
        y_predprob = []
        for batch in tqdm(self.dt.iter_batches(which=which, batch_size=self.model_args.eval_batch_size), ncols=80):
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
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            preds_label.extend(preds)
            gold_label.extend(batch[-1])
            y_predprob.extend(F.softmax(logits, dim=1).detach().cpu().numpy()[:, -1])
        preds_label = np.array(preds_label)
        gold_label = np.array(gold_label)
        acc = simple_accuracy(preds_label, gold_label)
        f1_micro = sklearn.metrics.f1_score(gold_label, preds_label, average="micro")
        print(f"{which} acc={acc}, f1_micro={f1_micro}")
        with open(os.path.join(self.data_args.save_results_path, f"{self.data_args.dataset_name}.jsonl"), "a") as f:
            f.write(f"Epoch-{epoch}: {which} acc={acc} f1_micro={f1_micro}\n")

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
    Trainer = TabFV_Trainer(
        model_args=model_args, 
        data_args=data_args, 
        device=device
    )
    Trainer.train()