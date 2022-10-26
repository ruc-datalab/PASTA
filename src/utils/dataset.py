import os
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from .args import ModelArguments, DataArguments
from .linearize import IndexedRowTableLinearize
from transformers import DebertaV2Tokenizer
from .entitylink import sub_func
from tqdm import tqdm
import pandas as pd
import torch

class DataManager:
    def __init__(self, data_args):
        #load tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(data_args.tokenizer_path)
        self.tokenizer.model_max_length = 512
        self.table_processor = IndexedRowTableLinearize()
        self.dataset_path = data_args.dataset_path
        self.dataset = data_args.dataset_name
        self.cache_file = data_args.cache_file
        if not os.path.exists(self.cache_file):
            os.mkdir(self.cache_file)
        
        self.prepare_data()

    def prepare_data(self):
        if self.dataset == 'tabfact':
            if os.path.exists(os.path.join(self.cache_file, 'train')):
                print(f"loading dataset from {self.cache_file}")
                train_smps = torch.load(os.path.join(self.cache_file, 'train'))
                test_smps = torch.load(os.path.join(self.cache_file, 'test'))
                valid_smps = torch.load(os.path.join(self.cache_file, 'val'))
                simple_smps = torch.load(os.path.join(self.cache_file, 'simple'))
                complex_smps = torch.load(os.path.join(self.cache_file, 'complex'))
            else:
                print(f"write dataset to {self.cache_file}")
                self.tables = self.load_the_tables()
                self.all_data = self.load_all_the_data_tabfact()
                train_smps = self.get_data_tabfact('train')
                test_smps = self.get_data_tabfact('test')
                valid_smps = self.get_data_tabfact('val')
                simple_smps = self.get_data_tabfact('simple')
                complex_smps = self.get_data_tabfact('complex')
            self.train_dataset = MyDataSet(train_smps)
            self.dev_dataset = MyDataSet(valid_smps)
            self.test_dataset = MyDataSet(test_smps)
            self.simple_smps = MyDataSet(simple_smps)
            self.complex_smps = MyDataSet(complex_smps)

        elif self.dataset == 'semtabfacts':
            if os.path.exists(os.path.join(self.cache_file, 'train')):
                print(f"loading dataset from {self.cache_file}")
                train_smps = torch.load(os.path.join(self.cache_file, 'train'))
                test_smps = torch.load(os.path.join(self.cache_file, 'test'))
                valid_smps = torch.load(os.path.join(self.cache_file, 'val'))
            else:
                print(f"write dataset to {self.cache_file}")
                self.tables = self.load_the_tables()
                train_smps = self.get_data_semtabfacts('train')
                test_smps = self.get_data_semtabfacts('test')
                valid_smps = self.get_data_semtabfacts('val')
            self.train_dataset = MyDataSet(train_smps)
            self.dev_dataset = MyDataSet(valid_smps)
            self.test_dataset = MyDataSet(test_smps)
            
        elif self.dataset == 'pasta':
            if os.path.exists(os.path.join(self.cache_file, 'train')):
                print(f"loading dataset from {self.cache_file}")
                train_smps = torch.load(os.path.join(self.cache_file, 'train'))
                valid_smps = torch.load(os.path.join(self.cache_file, 'val'))
            else:
                print(f"write dataset to {self.cache_file}")
                self.tables = self.load_the_tables()
                self.all_data = self.load_all_the_data_pasta()
                train_smps = self.get_data_pasta('train')
                valid_smps = self.get_data_pasta('val')
            self.train_dataset = MyDataSet(train_smps)
            self.dev_dataset = MyDataSet(valid_smps)
        else:
            raise NotImplementedError()

        
    def load_the_tables(self):
        if self.dataset == 'tabfact' or self.dataset == 'pasta':
            table_path = os.path.join(self.dataset_path, 'data/all_csv')
        elif self.dataset == 'semtabfacts':
            table_path = os.path.join(self.dataset_path, 'csv')
        else:
            raise NotImplementedError()
        print(f"load tables from {table_path}")
        tables = {}
        for fnm in tqdm(os.listdir(table_path)):
            tables[fnm] = self.load_a_table(os.path.join(table_path, fnm))
        return tables

    def load_a_table(self, fnm):
        table_contents = []
        if self.dataset == 'tabfact' or self.dataset == 'pasta':
            table_df = pd.read_csv(fnm, sep='#').astype(str)
        elif self.dataset == 'semtabfacts':
            table_df = pd.read_csv(fnm).astype(str)
        else:
            raise NotImplementedError()
        table_contents.append(table_df.columns.tolist()) #header
        table_contents.extend(table_df.values.tolist()) #rows
        return table_contents 
    
    def takeSecond(self, elem):
        return elem[1]
    
    def pasta_mask_tokens(self, sentence, cloze, linearized_table):
        sentence_toks = self.tokenizer.tokenize(sentence)
        cloze_toks = self.tokenizer.tokenize(cloze)
        cloze_extend = cloze.replace("[MASK]", "[MASK]"*(len(sentence_toks)-len(cloze_toks)+1))
        target_str = sentence + " " + linearized_table
        input_str = cloze_extend + " " + linearized_table
        label = self.tokenizer(target_str, padding="max_length", truncation=True, return_tensors="pt")['input_ids'][0]
        input = self.tokenizer(input_str, padding="max_length", truncation=True, return_tensors="pt")
        mask_ids = input['input_ids'][0]
        no_mask_ids = torch.nonzero(mask_ids != 128000)
        ignore_list = [int(i) for i in no_mask_ids]
        label[ignore_list] = -100
        return input["input_ids"][0], input["token_type_ids"][0], input["attention_mask"][0], label
    
    def get_data_pasta(self, which):
        if which == 'train':
            fnm = os.path.join(self.dataset_path, 'data/train_id.json')
            cache_fnm = os.path.join(self.cache_file,'train')
        elif which == 'val':
            fnm = os.path.join(self.dataset_path, 'data/val_id.json')
            cache_fnm = os.path.join(self.cache_file,'val')
        else:
            assert 1 == 2, "which should be in train/val"
        table_ids = eval(open(fnm).read())
        smps = []
        for tab_id in tqdm(table_ids):
            if tab_id not in self.all_data.keys():
                continue
            tab_data = self.all_data[tab_id]
            sentences, clozes = tab_data
            table = self.tables[tab_id]
            table_dict = {}
            table_dict["header"] = table[0]
            table_dict["rows"] = table[1:]
            linearized_table = self.table_processor.process_table(table_dict)
            for sentence, cloze in zip(sentences, clozes):
                input_ids, token_type_ids, attention_mask, label = self.pasta_mask_tokens(sentence, cloze, linearized_table)
                smps.append([input_ids, token_type_ids, attention_mask, label])
        print(f'{which} sample num={len(smps)}')
        torch.save(smps, cache_fnm)
        return smps
    
    def get_data_tabfact(self, which):
        if which == 'train':
            fnm = os.path.join(self.dataset_path, 'data/train_id.json')
            cache_fnm = os.path.join(self.cache_file,'train')
        elif which == 'val':
            fnm = os.path.join(self.dataset_path, 'data/val_id.json')
            cache_fnm = os.path.join(self.cache_file, 'val')
        elif which == 'test':
            fnm = os.path.join(self.dataset_path, 'data/test_id.json')
            cache_fnm = os.path.join(self.cache_file, 'test')
        elif which == 'simple':
            fnm = os.path.join(self.dataset_path, 'data/simple_test_id.json')
            cache_fnm = os.path.join(self.cache_file, 'simple')
        elif which == 'complex':
            fnm = os.path.join(self.dataset_path, 'data/complex_test_id.json')
            cache_fnm = os.path.join(self.cache_file, 'complex')
        else:
            assert 1 == 2, "which should be in train/val/test"
        table_ids = eval(open(fnm).read())
        smps = []
        self.stop = list(set(stopwords.words('english')))
        for tab_id in tqdm(table_ids):
            tab_dat = self.all_data[tab_id]
            qs, labels, caption = tab_dat
            table = self.tables[tab_id]
            table_head = table[0]
            table_body = table[1:]
            table_len = len(self.tokenizer.tokenize(self.table_processor.process_table({"header": table_head, "rows": table_body})))
            if table_len > 1000:
                need_trunc = True
            else:
                need_trunc = False
            for q,l in zip(qs,labels):
                table_dict = {}
                # col-select
                col_list = list(range(len(table_head)))               
                if need_trunc:
                    entity_link = sub_func(tab_id, '', q)
                    if entity_link == None:
                        entity_link = q
                    entity_index = [i for (i,j) in enumerate(entity_link) if j=='#']
                    entity_col_list = []
                    for i in range(0,len(entity_index)-1,2):
                        s = entity_link[entity_index[i] : entity_index[i+1]]
                        s = re.split("[;#]",s)[2]
                        s = s.split(",")
                        entity_col_list.append(int(s[1]))
                    entity_col_list = list(set(entity_col_list))
                    if len(entity_col_list) != len(col_list):
                        col_list.remove(random.choice([i for i in col_list if i not in entity_col_list]))
                # row-rank
                row_list = []
                q_tok = WordPunctTokenizer().tokenize(q)
                q_tok = [i for i in q_tok if i not in self.stop]
                row_scores = []
                table_body_after_select = []
                for idx, row in enumerate(table_body):
                    table_body_after_select.append([row[i] for i in col_list])
                    row_str = ' '.join([row[i] for i in col_list])
                    row_tok = WordPunctTokenizer().tokenize(row_str)
                    score = len(list(set(row_tok)&set(q_tok)))
                    row_scores.append((idx, score))
                row_scores.sort(key=self.takeSecond,reverse=True)
                row_list = [item[0] for item in row_scores]

                table_body_after_ranking = [table_body_after_select[i] for i in row_list]
                table_dict["header"] = [table_head[i] for i in col_list]
                table_dict["rows"] = table_body_after_ranking
                table_linearized = self.table_processor.process_table(table_dict)
                table_claim = q + " " + table_linearized
                input = self.tokenizer(table_claim, padding="max_length", truncation=True, return_tensors="pt")
                smps.append([input["input_ids"][0],input["token_type_ids"][0],input["attention_mask"][0],l])
        print(f'{which} sample num={len(smps)}')
        torch.save(smps, cache_fnm)
        return smps

    def get_data_semtabfacts(self, which):
        if which == 'train':
            fnm = os.path.join(self.dataset_path, 'tsv/train_man_set.tsv')
            cache_fnm = os.path.join(self.cache_file, 'train')
        elif which == 'val':
            fnm = os.path.join(self.dataset_path, 'tsv/dev_set.tsv')
            cache_fnm = os.path.join(self.cache_file, 'val')
        elif which == 'test':
            fnm = os.path.join(self.dataset_path, 'tsv/test_a_set.tsv')
            cache_fnm = os.path.join(self.cache_file, 'test')
        else:
            assert 1 == 2, "which should be in train/val/test"
        raw_data = pd.read_csv(fnm,sep="\t")
        smps = []
        self.stop = list(set(stopwords.words('english')))
        for index, item in tqdm(raw_data.iterrows()):
            q = item['question']
            tab_id = item['table_file']
            label = item['answer_text']
            if label not in [0, 1]: # we only verify entailed/refuted statements
                continue
            table = self.tables[tab_id]
            table_head = table[0]
            table_body = table[1:]            
            table_dict = {}       
            table_dict["header"] = table_head
            table_dict["rows"] = table_body
            table_linearized = self.table_processor.process_table(table_dict)
            table_claim = q + " " + table_linearized
            input = self.tokenizer(table_claim, padding="max_length", truncation=True, return_tensors="pt")
            smps.append([input["input_ids"][0],input["token_type_ids"][0],input["attention_mask"][0],label])
        print(f'{which} sample num={len(smps)}')
        torch.save(smps, cache_fnm)
        return smps

    def load_all_the_data_tabfact(self):
        all_data = {}
        for fnm in [os.path.join(self.dataset_path, "collected_data/r1_training_all.json"), os.path.join(self.dataset_path, "collected_data/r2_training_all.json")]:
            infos = eval(open(fnm).read())
            all_data.update(infos)
        return all_data
    
    def load_all_the_data_pasta(self):
        all_data = {}
        for fnm in tqdm(os.listdir(os.path.join(self.dataset_path, "raw_data/"))):
            infos = eval(open(os.path.join(self.dataset_path, "raw_data/", fnm)).read())
            all_data.update(infos)
        print(len(all_data))
        return all_data
    
    def iter_batches(self, which="train", samples=None, batch_size=None):
        if which == 'train':
            return DataLoader(shuffle=True, dataset=self.train_dataset, batch_size=batch_size)
        elif which == 'dev':
            return DataLoader(shuffle=False, dataset=self.dev_dataset, batch_size=batch_size)
        elif which == 'test':
            return DataLoader(shuffle=False, dataset=self.test_dataset, batch_size=batch_size)
        elif which == 'simple':
            return DataLoader(shuffle=False, dataset=self.simple_smps, batch_size=batch_size)
        elif which == 'complex':
            return DataLoader(shuffle=False, dataset=self.complex_smps, batch_size=batch_size)


class MyDataSet(Dataset):
    def __init__(self, smps):
        self.smps = smps
        super().__init__()

    def __getitem__(self, i):
        input_ids, token_type_ids, attention_mask, label= self.smps[i]
        return input_ids, token_type_ids, attention_mask, label
    def __len__(self):
        return len(self.smps)
