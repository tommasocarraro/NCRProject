"""
This module contains some utility functions for the preprocessing of the datasets reported in the paper.
Every dataset is different, so in this file there is one function per dataset. These functions load the dataset and
preprare it to make it compatible with the NCR framework. In particular, they convert user and item ids to unique
integer identifiers. In fact, in the Amazon datasets the user and item identifiers are strings. Also, these functions
force the indexes to start from 0. In fact, in some datasets (like MovieLens), the user and item identifiers start
from 1 instead of zero. Finally, the functions remove from the original datasets the fields that are not needed for
the training of the NCR model and return a pandas dataframe containing the following information for each user-item
interaction:
    - user id: the id of the user;
    - item id: the id of the item;
    - rating: the score gave by the user for the item (usually an integer between 1 and 5);
    - timestamp: the timestamp related to the moment in which the user reviewed the item.
"""

import pandas as pd
import json

def prepare_movielens_100k(path_to_file):
    """
    It prepares the MovieLens 100k dataset for the training with the NCR framework.
    :param path_to_file: a string containing the path to the MovieLens 100k dataset file.
    :return: a pandas dataframe with the structure explained in the header of this file.
    """
    dataset = pd.read_csv(path_to_file, sep='\t')
    dataset.columns = ["userID", "itemID", "rating", "timestamp"]

    dataset['userID'] -= 1  # convert user IDs in indexes starting from 0
    dataset['itemID'] -= 1  # convert item IDs in indexes starting from 0

    return dataset

def prepare_amazon(path_to_file):
    """
    It prepares the Amazon (Movies and TV, or Electronics) dataset for the training with the NCR framework.
    :param path_to_file: a string containing the path to the Amazon (Movies and TV, or Electronics) dataset file.
    :return: a pandas dataframe with the structure explained in the header of this file.
    """
    dataset_dict = {"userID": [], "itemID": [], "rating": [], "timestamp": []}
    user_to_id = {}
    item_to_id = {}
    item_id, user_id = 0, 0
    with open(path_to_file, 'r') as f:
        line = f.readline()
        while line:
            json_line = json.loads(line)
            # this is needed to convert the string ids into numeric ids starting from 0
            if json_line['reviewerID'] not in user_to_id:
                user_to_id[json_line['reviewerID']] = user_id
                user_id += 1
            if json_line['asin'] not in item_to_id:
                item_to_id[json_line['asin']] = item_id
                item_id += 1
            dataset_dict['userID'].append(user_to_id[json_line['reviewerID']])
            dataset_dict['itemID'].append(item_to_id[json_line['asin']])
            dataset_dict['rating'].append(int(json_line['overall']))
            dataset_dict['timestamp'].append(int(json_line['unixReviewTime']))
            line = f.readline()

    dataset = pd.DataFrame.from_dict(dataset_dict)
    return dataset