import argparse
from ncr.data import Dataset
from ncr.samplers import DataSampler
from ncr.nets import NCR
from ncr.models import NCRTrainer
from ncr.evaluation import ValidFunc, logic_evaluate
import torch
import pandas as pd
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set pytorch device for computation

def main():
    """
    This is the main() function which runs the experiments on the NCR framework. An experiment is made of the following
    steps:
        - preprocessing of the specified dataset with the specified parameters. This includes the split of the dataset
        into train, validation, and test folds;
        - construction of the NCR neural architecture with the specified parameters;
        - train of the NCR model with the specified training details and validation details;
        - test of the NCR model with the specified test metrics.
    """
    # init args
    init_parser = argparse.ArgumentParser(description='Experiment')
    init_parser.add_argument('--threshold', type=int, default=4,
                             help='Threshold for generating positive/negative feedback.')
    init_parser.add_argument('--order', type=bool, default=True,
                             help='Flag indicating whether the ratings have to be ordered by timestamp on not.')
    init_parser.add_argument('--leave_n', type=int, default=1,
                             help='Number of positive interactions that are hold out from each user for validation and test sets.')
    init_parser.add_argument('--keep_n', type=int, default=5,
                             help='Minimum number of positive interactions that are kept in training set for each user.')
    init_parser.add_argument('--max_history_length', type=int, default=5,
                             help='Maximum length of history for each interaction (i.e. maximum number of items at the left of the implication in the logical expressions)')
    init_parser.add_argument('--n_neg_train', type=int, default=1,
                             help='Number of negative items randomly sampled for each training interaction. The items are sampled from the set of items that the user has never seen.')
    init_parser.add_argument('--n_neg_val_test', type=int, default=100,
                             help='Number of negative items randomly sampled for each validation/test interaction. The items are sampled from the set of items that the user has never seen.')
    init_parser.add_argument('--training_batch_size', type=int, default=128,
                             help='Size of training set batches.')
    init_parser.add_argument('--val_test_batch_size', type=int, default=128 * 2,
                             help='Size of validation/test set batches.')
    init_parser.add_argument('--seed', type=int, default=2022,
                             help='Random seed to reproduce the experiments.')
    init_parser.add_argument('--emb_size', type=int, default=64,
                             help='Size of users, item, and event embeddings.')
    init_parser.add_argument('--dropout', type=float, default=0.0,
                             help='Percentage of units that are randomly shut down in every hidden layer during training.')
    init_parser.add_argument('--lr', type=float, default=0.001,
                             help='Learning rate for the training of the model.')
    init_parser.add_argument('--l2', type=float, default=0.0001,
                             help='Weight for the L2 regularization.')
    init_parser.add_argument('--r_weight', type=float, default=0.1,
                             help='Weight for the logical regularization.')
    init_parser.add_argument('--val_metric', type=str, default='ndcg@5',
                             help='Metric computed for the validation of the model.')
    init_parser.add_argument('--test_metrics', type=list, default=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'],
                             help='Metrics computed for the test of the model.')
    init_parser.add_argument('--n_epochs', type=int, default=100,
                             help='Number of epochs for the training of the model.')
    init_parser.add_argument('--early_stop', type=int, default=5,
                             help='Number of epochs for early stopping. It should be > 1.')
    init_parser.add_argument('--at_least', type=int, default=20,
                             help='Minimum number of epochs before starting with early stopping.')
    init_parser.add_argument('--save_load_path', type=str, default="saved-models/best_model.json",
                             help='Path where the model has to be saved during training. The model is saved every time the validation metric increases. This path is also used for loading the best model before the test evaluation.')
    init_parser.add_argument('--n_times', type=int, default=10,
                             help='Number of times the test evaluation is performed (metrics are averaged across these n_times evaluations). This is required since the negative items are randomly sampled.')
    init_parser.add_argument('--dataset', type=str, default="movielens_100k",
                             help="Dataset on which the experiment has to be performed ('movielens_100k', 'amazon_movies_tv', 'amazon_electronics').")
    init_parser.add_argument('--test_only', type=bool, default=False,
                             help='Flag indicating whether it has to be computed only the test evaluation or not. If True, there should be a model checkpoint to load in the specified save path.')
    init_parser.add_argument('--premise_threshold', type=int, default=0,
                             help='Threshold for filtering logical expressions based on the number of premises. All the logical expressions with a number of premises equal to or lower than premise_threshold are removed from the dataset.')
    init_args, init_extras = init_parser.parse_known_args()

    # take the correct dataset
    if init_args.dataset == "movielens_100k":
        raw_dataset = pd.read_csv("datasets/movielens-100k/movielens_100k.csv")
    if init_args.dataset == "amazon_movies_tv":
        raw_dataset = pd.read_csv("datasets/amazon-movies-tv/movies_tv.csv")
    if init_args.dataset == "amazon_electronics":
        raw_dataset = pd.read_csv("datasets/amazon-electronics/electronics.csv")

    # create train, validation, and test sets
    dataset = Dataset(raw_dataset)
    dataset.process_data(threshold=init_args.threshold, order=init_args.order, leave_n=init_args.leave_n,
                         keep_n=init_args.keep_n, max_history_length=init_args.max_history_length,
                         premise_threshold=init_args.premise_threshold)
    if not init_args.test_only:
        train_loader = DataSampler(dataset.train_set, dataset.user_item_matrix, n_neg_samples=init_args.n_neg_train,
                                   batch_size=init_args.training_batch_size, shuffle=True, seed=init_args.seed,
                                   device=device)
        val_loader = DataSampler(dataset.validation_set, dataset.user_item_matrix,
                                 n_neg_samples=init_args.n_neg_val_test, batch_size=init_args.val_test_batch_size,
                                 shuffle=False, seed=init_args.seed, device=device)

    test_loader = DataSampler(dataset.test_set, dataset.user_item_matrix, n_neg_samples=init_args.n_neg_val_test,
                              batch_size=init_args.val_test_batch_size, shuffle=False, seed=init_args.seed,
                              device=device)

    ncr_net = NCR(dataset.n_users, dataset.n_items, emb_size=init_args.emb_size, dropout=init_args.dropout,
                  seed=init_args.seed).to(device)

    ncr_model = NCRTrainer(ncr_net, learning_rate=init_args.lr, l2_weight=init_args.l2,
                           logic_reg_weight=init_args.r_weight)

    if not init_args.test_only:
        ncr_model.train(train_loader, valid_data=val_loader, valid_metric=init_args.val_metric,
                        valid_func=ValidFunc(logic_evaluate), num_epochs=init_args.n_epochs,
                        at_least=init_args.at_least, early_stop=init_args.early_stop,
                        save_path=init_args.save_load_path, verbose=1)

    ncr_model.load_model(init_args.save_load_path)

    ncr_model.test(test_loader, test_metrics=init_args.test_metrics, n_times=init_args.n_times)

if __name__ == '__main__':
    main()