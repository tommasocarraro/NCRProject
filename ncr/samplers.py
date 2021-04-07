import numpy as np
import torch
import random

__all__ = ['Sampler', 'DataSampler']

class Sampler(object):
    """Sampler base class.
    A sampler is meant to be used as a generator of batches useful in training neural networks.
    Notes
    -----
    Each new sampler must extend this base class implementing all the abstract
    special methods.
    """
    def __init__(self, *args, **kargs):
        pass

    def __len__(self):
        """Return the number of batches.
        """
        raise NotImplementedError

    def __iter__(self):
        """Iterate through the batches yielding a batch at a time.
        """
        raise NotImplementedError


class DataSampler(Sampler):
    """This is a standard sampler that returns batches without any particular constraint.
    Bathes are randomly returned with the defined dimension (i.e., ``batch_size``). If ``shuffle``
    is set to ``False`` then the sampler returns batches with the same order as in the original
    dataset.
    Parameters/Attributes
    ----------
    data : the dataset fold for which the batches have to be created.
    n_neg_samples: number of negative samples that have to be generated for each interaction (during training should be
    one, while in validation and test should be 100 (these are the specs reported in the paper))
    user_item_matrix: this is a scipy sparse matrix containing the user-item interactions in the dataset. This is needed
    for the sampling of negative items.
    batch_size : the size of the batches, by default 1.
    shuffle : whether the data set must by randomly shuffled before creating the batches, by default ``True``
    seed: the random seed for the shuffle
    device: the device where the torch tensors have to be put
    remove_one_premise: flag indicating whether the logical expressions with only one premise have to be considered
    during training or not
    """
    def __init__(self,
                 data,
                 user_item_matrix,
                 n_neg_samples=1,
                 batch_size=128,
                 shuffle=True,
                 seed=2022,
                 device='cuda:0',
                 remove_one_premise=False):
        super(DataSampler, self).__init__()
        self.data = data
        self.user_item_matrix = user_item_matrix
        self.n_neg_samples = n_neg_samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.device = device
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.remove_one_premise = remove_one_premise

    def __len__(self):
        # here, it is not sufficient to return int(np.ceil(self.data.shape[0] / self.batch_size)) since we have
        # grouped the dataset into small datasets each of one correspond to a history length
        dataset_size = 0
        length = self.data.groupby("history_length")  # histories could be of different lengths, so we need to group
        # histories of the same length in the same batch
        for i, (_, l) in enumerate(length):
            if self.remove_one_premise and i == 0:  # on the first index there are the logical expressions with only one
                # premise, that are the logical expressions that we do not have to consider
                continue
            dataset_size += int(np.ceil(l.shape[0] / self.batch_size))
        return dataset_size

    def __iter__(self):
        # for each epoch, each positive interaction has a random sampled negative interaction for the pair-wise learning
        # instead, in validation, each positive interaction has 100 random sampled negative interactions for the
        # computation of the metrics
        length = self.data.groupby("history_length")  # histories could be of different lengths, so we need to group
        # histories of the same length in the same batch

        for i, (_, l) in enumerate(length):
            if self.remove_one_premise and i == 0:
                continue
            # get numpy arrays of the dataframe fields that we need to train the model
            group_users = np.array(list(l['userID']))
            group_items = np.array(list(l['itemID']))
            group_histories = np.array(list(l['history']))
            group_feedbacks = np.array(list(l['history_feedback']))

            n = group_users.shape[0]
            idxlist = list(range(n))
            if self.shuffle: # every small dataset based on history length is shuffled before preparing batches
                np.random.shuffle(idxlist)

            for _, start_idx in enumerate(range(0, n, self.batch_size)):
                end_idx = min(start_idx + self.batch_size, n)
                batch_users = torch.from_numpy(group_users[idxlist[start_idx:end_idx]])
                batch_items = torch.from_numpy(group_items[idxlist[start_idx:end_idx]])
                batch_histories = torch.from_numpy(group_histories[idxlist[start_idx:end_idx]])
                batch_feedbacks = torch.from_numpy(group_feedbacks[idxlist[start_idx:end_idx]])

                # here, we generate negative items for each interaction in the batch
                batch_user_item_matrix = self.user_item_matrix[batch_users].toarray()  # this is the portion of the
                # user-item matrix for the users in the batch
                batch_user_unseen_items = 1 - batch_user_item_matrix  # this matrix contains the items that each user
                # in the batch has never seen
                negative_items = []  # this list contains a list of negative items for each interaction in the batch
                for u in range(batch_users.size(0)):
                    u_unseen_items = batch_user_unseen_items[u].nonzero()[0]  # items never seen by the user
                    # here, we generate n_neg_samples indexes and use them to take n_neg_samples random items from
                    # the list of the items that the user has never seen
                    rnd_negatives = u_unseen_items[random.sample(range(u_unseen_items.shape[0]), self.n_neg_samples)]
                    # we append the list to negative_items
                    negative_items.append(rnd_negatives)
                batch_negative_items = torch.tensor(negative_items)

                yield batch_users.to(self.device), batch_items.to(self.device), batch_histories.to(self.device), \
                      batch_feedbacks.to(self.device), batch_negative_items.to(self.device)