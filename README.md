# NCRProject
This repository contains my implementation of Neural Collaborative Reasoning (NCR), a paper published at WWW '21 which integrates the power of neural networks and logical reasoning in the top-N recommendation task. This is the [link](https://arxiv.org/pdf/2005.08129.pdf) to the paper. It is kindly suggested to read the paper before going through the code and its execution. 

## Reference
For inquiries on the paper or original [code](https://github.com/rutgerswiselab/NCR) contact Hanxiong Chen (hanxiong.chen@rutgers.edu) or Yongfeng Zhang (yongfeng.zhang@rutgers.edu). 

The full paper reference is the following:
Hanxiong Chen, Shaoyun Shi, Yunqi Li, Yongfeng Zhang. 2021. Neural Collaborative Reasoning. In Proceedings of the Web Conference 2021(WWW ’21)

## Enviroments
Python version: 3.9

Packages:

bottleneck==1.3.2

numpy==1.20.1

torch==1.8.0

pandas==1.2.3

scipy==1.6.1

## Repository organization
The repository is organized with the following folders:
1. commands: it contains the commands (explained below) to run the experiments reported in the original paper for the NCR model;
2. datasets: it contains three CSV files, one for each dataset (MovieLens 100k, Amazon Movies and TV, Amazon Electronics). Each row of these files is composed of the following fields: userID, itemID, rating (from 1 to 5), timestamp (review timestamp);
3. ncr: it contains the NCR framework (explained below);
4. results: it contains the results obtained for the NCR model with the execution of the commands reported in "commands". These results are composed of the metrics ndcg@5, ndcg@10, hr@5, hr@10 computed on the test set of the three datasets reported in the paper (MovieLens 100k, Amazon Movies and TV, Amazon Electronics);
5. saved-models: it contains the pytorch checkpoints of the best models (best ndcg@5 on validation) obtained during training. These are the so-called pre-trained models. For the MovieLens 100k dataset it has been possible to directly upload the model on GitHub. Since for the other two datasets the file is too big, it hes been provided a Google Drive link to the best models;
6. training-log: it contains one example of training log. It is the log of the training of NCR on the MovieLens 100k dataset using the first command in the /commands/commands.txt file;
7. tutorials: it contains an initial tutorial of NCR. The tutorial firstly presents a brief but detailed summary of the paper Neural Collaborative Reasoning, then it explains how to run a simple experiment on the MovieLens 100k dataset using the code provided in this repository.

## NCR framework
The NCR framework is composed of the modules explained below. This is the structure used in [rectorch](https://github.com/makgyver/rectorch), a state-of-the-art recommender systems framework. You will find this model in the rectorch framework very soon.
1. data: this module contains functions for the preprocessing of the datasets. In particular, this module prepares the datasets for the NCR framework and split them in train, validation and test set as reported in the original paper. What does it mean that this module prepares the datasets for the NCR framework? First of all, the module sorts the dataset by timestamp and filters it based on the rating information. Rating equal to or higher than 4 are map to 1 (positive feedback), while ratings lower than 4 are map to 0 (negative feedback). Then, for each positive user-item interaction (positive feedback) in the dataset, it creates an "history" useful to build a logical expression. The history is a list of up to 5 items interacted by the user just before the selected positive interaction. It is suggested to carefully read the original paper to understand well this process;
2. evaluation: this module contains functions to perform different types of evaluations. In particular, the NCR model uses the logic_evaluate() function which performs the logical evaluation carefully explained in the original paper. As a recap, for each positive expression in the test set, 100 negative items are sampled from the items that the user has never seen in order to build 100 negative logical expressions. Then, the test metrics are computed based on the position of the positive item against the 100 negative items in the ranking output by the model. In particular, the NCR model outputs a prediction for each item which is computed as the cosine similarity between the item's logical expression event vector and the fixed TRUE vector. Higher the similarity between the logical expression and the TRUE vector, higher the recommendation score for the item and higher its position in the ranking. 
3. metrics: this module contains the code to compute some recommendation metrics. In particular, the NCR framework computes nDCG@k and Hit Ratio (HR@k);
4. models: this module contains the code for training and testing the NCR model (details about the training are presented in the original paper);
5. nets: this module contains the NCR neural architecture. The architecture is modular and adapts to the logical expression given in input (details about the neural architecture are presented in the orginal paper);
6. samplers: this module contains the data loaders. The data loaders load the dataset folds and split them in batches useful for an efficient training of the model;
7. utils: this module contains some utility functions to convert the original raw dataset files in CSV files ready for the NCR framework. The repository already contains the CSV files in the "datasets" folder.

## How to run NCR experiments
As explained before, in the "commands" folder there are some commands to train and test the NCR model on the three different datasets.
These commands contains only few parameters since the others have a default value. The full list is reported below:
1. --threshold: Threshold for generating positive/negative feedback. Default=4;
2. --order: Flag indicating whether the ratings have to be ordered by timestamp on not. Default=True;
3. --leave_n: Number of positive interactions that are hold out from each user for validation and test sets. Default=1;
4. --keep_n: Minimum number of positive interactions that are kept in training set for each user. Default=5;
5. --max_history_length: Maximum length of history for each interaction (i.e. maximum number of items at the left of the implication in the logical expressions. Default=5;
6. --n_neg_train: Number of negative items randomly sampled for each training interaction. The items are sampled from the set of items that the user has never seen. Default=1;
7. --n_neg_val_test: Number of negative items randomly sampled for each validation/test interaction. The items are sampled from the set of items that the user has never seen. Default=100;
8. --training_batch_size: Size of training set batches. Default=128;
9. --val_test_batch_size: Size of validation/test set batches. Default=256;
10. --seed: Random seed to reproduce the experiments. Default=2022;
11. --emb_size: Size of users, item, and event embeddings. Default=64;
12. --dropout: Percentage of units that are randomly shut down in every hidden layer during training. Default=0.0;
13. --lr: Learning rate for the training of the model. Default: 0.001;
14. --l2: Weight for the L2 regularization. Default: 0.0001;
15. --r_weight: Weight for the logical regularization. Default: 0.1;
16. --val_metric: Metric computed for the validation of the model. Default='ndcg@5';
17. --test_metrics: Metrics computed for the test of the model. Default=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'];
18. --n_epochs: Number of epochs for the training of the model. Default=100;
19. --early_stop: Number of epochs for early stopping. It should be > 1. Default=5;
20. --at_least: Minimum number of epochs before starting with early stopping. Default=20;
21. --save_load_path: Path where the model has to be saved during training. The model is saved every time the validation metric increases. This path is also used for loading the best model before the test evaluation. Default="saved-models/best_model.json";
22. --n_times: Number of times the test evaluation is performed (metrics are averaged across these n_times evaluations). This is required since the negative items are randomly sampled. Default=10;
23. --dataset: Dataset on which the experiment has to be performed ('movielens_100k', 'amazon_movies_tv', 'amazon_electronics'). Default="movielens_100k";
24. --test_only: Flag indicating whether it has to be computed only the test evaluation or not. If True, there should be a model checkpoint to load in the specified save path. Default=False;
25. --premise_threshold: Threshold for filtering logical expressions based on the number of premises (number of propositional variables at the left side of the implication). All the logical expressions with a number of premises equal to or lower than premise_threshold are removed from the dataset before the training of the model.
