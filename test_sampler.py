from ncr.data import Dataset
from ncr.samplers import DataSampler
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # set pytorch device for computation

if __name__ == '__main__':
    dataset = Dataset("datasets/movielens-100k/u.data")

    dataset.process_data(threshold=4, order=True)

    train_loader = DataSampler(dataset.train_set, dataset.user_item_matrix, batch_size=128, device=device)

    for batch_idx, batch_data in enumerate(train_loader):
        print(batch_data)
        break
