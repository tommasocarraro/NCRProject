from ncr.data import Dataset
from ncr.samplers import DataSampler
from ncr.nets import NCR
from ncr.models import NCRTrainer
from ncr.evaluation import ValidFunc, evaluate
import torch
import numpy as np
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set pytorch device for computation

if __name__ == '__main__':
    save_path = "saved-models/best_ncr_model.json"
    dataset = Dataset("datasets/movielens-100k/u.data")

    dataset.process_data(threshold=4, order=True, leave_n=1, keep_n=5, max_history_length=5)

    train_loader = DataSampler(dataset.train_set, dataset.user_item_matrix, n_neg_samples=1, batch_size=128,
                               shuffle=True, seed=2022, device=device)
    val_loader = DataSampler(dataset.validation_set, dataset.user_item_matrix, n_neg_samples=100, batch_size=200,
                             shuffle=False, seed=2022, device=device)
    test_loader = DataSampler(dataset.test_set, dataset.user_item_matrix, n_neg_samples=100, batch_size=200,
                             shuffle=False, seed=2022, device=device)

    ncr_net = NCR(dataset.n_users, dataset.n_items, emb_size=64, dropout=0.0, seed=2022).to(device)

    model = NCRTrainer(ncr_net, learning_rate=0.001, l2_weight=0.0001, logic_reg_weight=0.01)

    model.train(train_loader, valid_data=val_loader, valid_metric='ndcg@5', valid_func=ValidFunc(evaluate),
                num_epochs=100, early_stop=5, save_path=save_path, verbose=1)

    model.load_model(save_path)

    model.test(test_loader, metric_list=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'], n_times=10)