# Introduction
This repository contains source code for the PASTA model, a pre-trained language model for table-based fact verification. The full paper is accepted to EMNLP2022 "Table-Operations Aware Fact Verification via Sentence-Table Cloze Pre-training". The code will be released in November.

## Requirements
Before running the code, please make sure your Python version is above **3.8**. Then install the necessary packages by:
```sh
pip install -r requirements.txt
```

## Datasets Preparation

### TabFact
Please download the TabFact dataset from [the official GitHub repository](https://github.com/wenhuchen/Table-Fact-Checking) and put it under the folder "PASTA/datasets".
```sh
git clone git@github.com:wenhuchen/Table-Fact-Checking.git
mv Table-Fact-Checking tabfact
mv tabfact PASTA/datasets
```

### SEM-TAB-FACTS
Please download the sem-tab-facts dataset from the official website: 
- [Manually annotated training set](https://drive.google.com/file/d/1yObzEEZJ8qM7ZjrMcbtKZ-jofpL820ft/view)
- [Dev set](https://drive.google.com/file/d/1l5iojO8q_CB-sDCjlUpa7wVi8XUrqlss/view)
- [Test set](https://drive.google.com/file/d/1Trfq0Zd2tcAV4JIR9puopmy6NC1lMj5S/view)

Then refer to this [repository](https://github.com/devanshg27/sem-tab-fact) for standardizing the table header, or you can directly download the [dataset](https://drive.google.com/file/d/1iQ9y3UetDq0-Ib70us2Oo-pwx63U2rls/view) we have processed.

Finally, put the processed dataset under the folder "PASTA/datasets", and name it "semtabfacts".

## Pre-training

## Fine-tuning

### Run TabFact
Fine-tune on the Tabfact dataset with the following command. 
```sh
python src/run_finetune.py src/scripts/train_tabfact.json
```
Note that you need to modify the paths in the `.json` file to your own paths.

### Run SEM-TAB-FACTS
Following [LKA](https://aclanthology.org/2022.coling-1.120.pdf), we also use the trained model on the TabFact to initialize the training of SEM-TAB-FACTS. Therefore, You need to train on the TabFact dataset to get the checkpoint, or you can directly download the checkpoint we provide and put it under `save_checkpoints`.
Then, fine-tune on the SEM-TAB-FACTS dataset with the following command.
```sh
python src/run_finetune.py src/scripts/train_semtabfacts.json
```
Note that you need to modify the paths in the `.json` file to your own paths.

## Reference
