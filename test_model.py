from ncr.data import Dataset
from ncr.samplers import DataSampler
from ncr.nets import NCR
from ncr.models import NCRTrainer
from ncr.evaluation import ValidFunc, logic_evaluate
from ncr.utils import prepare_movielens_100k, prepare_amazon
import torch
import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # set pytorch device for computation

if __name__ == '__main__':

    save_path = "saved-models/best_movielens_100k.json"

    raw_dataset = prepare_movielens_100k("datasets/movielens-100k/u.data")
    #raw_dataset = pd.read_csv("datasets/amazon_electronics/electronics.csv")

    dataset = Dataset(raw_dataset)

    dataset.process_data(threshold=4, order=True, leave_n=1, keep_n=5, max_history_length=5)

    test_loader = DataSampler(dataset.test_set, dataset.user_item_matrix, n_neg_samples=100, batch_size=200,
                             shuffle=False, seed=2022, device=device)

    ncr_net = NCR(dataset.n_users, dataset.n_items, emb_size=64, dropout=0.0, seed=2022).to(device)

    model = NCRTrainer(ncr_net, learning_rate=0.001, l2_weight=0.0001, logic_reg_weight=0.1)

    model.load_model(save_path)

    model.test(test_loader, test_metrics=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'], n_times=10)