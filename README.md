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
Then refer to this [repository](https://github.com/devanshg27/sem-tab-fact) for standardizing the table header, or you can directly download the [dataset] we have processed.
Finally, put the processed dataset under the folder "PASTA/datasets", and name it "semtabfacts".

## Pre-training

## Fine-tuning

## Reference
