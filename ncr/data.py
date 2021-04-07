import pandas as pd
from scipy import sparse

__all__ = ['Dataset']

class Dataset(object):
    """This class manages a recommendation system dataset.
    It contains the information about the dataset, such as the number of users and items.
    It contains the methods for the preprocessing and split of the dataset.
    Parameters
    ----------
    dataset: this is a pandas dataframe containing the user-item interactions before preprocessing
    n_users: the number of users in the dataset
    n_items: the number of items in the dataset
    proc_dataset: this is a pandas dataframe containing the user-item interactions after preprocessing
    """

    def __init__(self, raw_dataset):
        """
        It initializes a Dataset object, computes the user item sparse matrix and dataset information.
        :param raw_dataset: pandas dataframe containing the user-item interactions of the dataset. This dataframe must
        have the following structure:
            - user id: the id of the user;
            - item id: the id of the item;
            - rating: the score gave by the user for the item (usually an integer between 1 and 5);
            - timestamp: the timestamp related to the moment in which the user reviewed the item.
        In the utils module there are functions that automatically create this structure based on the given dataset.
        """
        self.dataset = raw_dataset

        self.n_users = self.dataset['userID'].nunique()
        self.n_items = self.dataset['itemID'].nunique()

        self.user_item_matrix = self.compute_sparse_matrix()


    def compute_sparse_matrix(self):
        """
        It computes the user-item sparse matrix. Every row is a user and every column is an item.
        A 1 in the matrix means that the user liked the item, while a 0 means that the user disliked the item.
        :return: the scipy sparse user-item matrix representing the dataset interactions
        """
        group = self.dataset.groupby("userID")
        rows, cols = [], []
        values = []
        for i, (_, g) in enumerate(group):
            u = list(g['userID'])[0]  # user id
            items = set(list(g['itemID']))  # items on the history
            rows.extend([u] * len(items))
            cols.extend(list(items))
            values.extend([1] * len(items))
        return sparse.csr_matrix((values, (rows, cols)), (self.n_users, self.n_items))

    def process_data(self, threshold=4, order=True, leave_n=1, keep_n=5, max_history_length=5, premise_threshold=0):
        """
        It processes the dataset given the preprocessing parameters. In particular, it filters the user-item
        interactions using the threshold and orders them by timestamp field (if order is set to True). Ratings equal
        to or higher than threshold are converted to 1 (positive feedback), while ratings lower than threshold are
        converted to 0 (negative feedback). After this procedure, it creates train, validation and test folds as
        reported in the paper. For doing that, it calls leave_one_out_by_time(). Finally, it adds to the folds the
        information to generate the logical expressions for the training/testing of the model. For doing so, it calls
        generate_histories().
        :param threshold: the threshold used to filter the user-item ratings
        :param order: a flag indicating whether the dataset has to be ordered by timestamp or not
        :param leave_n: see leave_out_out_by_time()
        :param keep_n: see leave_out_out_by_time()
        :param max_history_length: see generate_histories()
        :param premise_threshold: see generate_histories(
        """
        # filter ratings by threshold
        self.proc_dataset = self.dataset.copy()
        self.proc_dataset['rating'][self.proc_dataset['rating'] < threshold] = 0
        self.proc_dataset['rating'][self.proc_dataset['rating'] >= threshold] = 1

        if order:
            self.proc_dataset = self.proc_dataset.sort_values(by=['timestamp', 'userID', 'itemID']).reset_index(drop=True)

        self.leave_out_out_by_time(leave_n, keep_n)
        self.generate_histories(max_hist_length=max_history_length, premise_threshold=premise_threshold)


    def leave_out_out_by_time(self, leave_n=1, keep_n=5):
        """
        It generates train, validation, and test folds of the dataset using the procedure reported in the paper.
        The procedure starts with the dataset ordered by timestamp.
        In particular:
            - the first keep_n positive interactions of each user are put in training set;
            - the last leave_n positive interactions of each user are held out for test set;
            - the second to the last leave_n interactions of each user are held out for validation set.
        :param leave_n: number of items that are left in validation and test set.
        :param warm_n: minimum number of positive interactions to leave in training dataset for each user.
        """

        train_set = []
        # generate training set by looking for the first keep_n POSITIVE interactions
        processed_data = self.proc_dataset.copy()
        for uid, group in processed_data.groupby('userID'):  # group by uid
            found, found_idx = 0, -1
            for idx in group.index:
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= keep_n:
                        break
            if found_idx > 0:
                train_set.append(group.loc[:found_idx + 1])
        train_set = pd.concat(train_set)
        # drop the training data info
        processed_data = processed_data.drop(train_set.index)

        # generate test set by looking for the last leave_n POSITIVE interactions
        test_set = []
        for uid, group in processed_data.groupby('userID'):
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            if found_idx > 0:
                test_set.append(group.loc[found_idx:])
        test_set = pd.concat(test_set)
        processed_data = processed_data.drop(test_set.index)

        validation_set = []
        for uid, group in processed_data.groupby('userID'):  #
            found, found_idx = 0, -1
            for idx in reversed(group.index):
                if group.loc[idx, 'rating'] > 0:
                    found_idx = idx
                    found += 1
                    if found >= leave_n:
                        break
            # put all the negative interactions encountered during the search process into validation set
            if found_idx > 0:
                validation_set.append(group.loc[found_idx:])
        validation_set = pd.concat(validation_set)
        processed_data = processed_data.drop(validation_set.index)

        # The remaining data (after removing validation and test) are all in training data
        self.train_set = pd.concat([train_set, processed_data])
        self.validation_set, self.test_set = validation_set.reset_index(drop=True), test_set.reset_index(drop=True)

    def generate_histories(self, max_hist_length=5, premise_threshold=0):
        """
        Generate history interaction sequence (items at the left side of implication) for each interaction in train,
        validation, and test sets, and appends it to the dataframe.
        In particular, it adds to the dataframe three columns:
            - history column: it contains the items to be put at the left side of implication;
            - history_feedback column: it contains the feedback for the items in the history;
            - history_length column: it contains the length of the history. This is needed for creating batches.
            In particular, we force each batch to have only samples with the same history length.
        :param max_hist_length: the max history length to keep (max number of items at the left side of the
        implication), ==0 value means keeps all.
        :param premise_threshold: it specifies a threshold for filtering logical expressions based on
        the number of premises. Specifically, all the logical expressions with a number of premises equal to or lower
        than premise_threshold will be removed from the dataset. The value should be between 0 (no filter) and
        max_hist_length - 1 (maximum applicable filter).
        For example, if we have logical expressions: a -> b, a ∧ b -> c and a ∧ b ∧ c -> d, and the parameter
        premise_threshold is set to 2, the first two expressions will be removed from the dataset.
        """
        history_dict = {} # it contains for each user the list of all the items he has seen
        feedback_dict = {} # it contains for each user the list of feedbacks he gave to the items he has seen
        for df in [self.train_set, self.validation_set, self.test_set]:
            history = [] # each element of this list is a list containing the history items of a single interaction
            fb = [] # each element of this list is a list containing the feedback for the history items of a
            # single interaction
            hist_len = [] # each element of this list indicates the number of history items of a single interaction
            uids, iids, feedbacks = df['userID'].tolist(), df['itemID'].tolist(), df['rating'].tolist()
            for i, uid in enumerate(uids):
                iid, feedback = iids[i], feedbacks[i]

                if uid not in history_dict:
                    history_dict[uid] = []
                    feedback_dict[uid] = []

                # list containing the history for current interaction
                tmp_his = history_dict[uid] if max_hist_length == 0 else history_dict[uid][-max_hist_length:]
                # list containing the feedbacks for the history of current interaction
                fb_his = feedback_dict[uid] if max_hist_length == 0 else feedback_dict[uid][-max_hist_length:]

                history.append(tmp_his)
                fb.append(fb_his)
                hist_len.append(len(tmp_his))

                history_dict[uid].append(iid)
                feedback_dict[uid].append(feedback)

            df['history'] = history
            df['history_feedback'] = fb
            df['history_length'] = hist_len

        # filtering logical expressions based on number of premises
        # we remove from the dataset all the logical expressions with a number of premises equal to or lower than
        # premise_threshold
        self.train_set = self.train_set[self.train_set.history_length > premise_threshold]
        self.validation_set = self.validation_set[self.validation_set.history_length > premise_threshold]
        self.test_set = self.test_set[self.test_set.history_length > premise_threshold]

        self.clean_data()


    def clean_data(self):
        """
        It removes all the interactions for which is not possible to construct a logical expression.
        In particular, it removes from train, validation and test sets those interactions (items at the right side of
        implication) that have a negative feedback. In fact, we want to predict only the positive items.
        Then, it removes all the interactions that have an empty history (the left side of implication would be empty).
        """
        self.train_set = self.train_set[self.train_set['rating'] > 0].reset_index(drop=True)
        self.train_set = self.train_set[self.train_set['history_feedback'].map(len) > 0].reset_index(drop=True)
        self.validation_set = self.validation_set[self.validation_set['rating'] > 0].reset_index(drop=True)
        self.validation_set = self.validation_set[self.validation_set['history_feedback'].map(len) > 0].reset_index(drop=True)
        self.test_set = self.test_set[self.test_set['rating'] > 0].reset_index(drop=True)
        self.test_set = self.test_set[self.test_set['history_feedback'].map(len) > 0].reset_index(drop=True)