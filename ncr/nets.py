import torch
import numpy as np
from torch.nn.init import normal_ as normal_init
import torch.nn.functional as F

__all__ = ['NCR']

class NCR(torch.nn.Module):
    """
    This is the class that implements the NCR neural architecture. This is only the architecture of the model.
    Utilities for the training of the model are put in the model module.
    :param n_users: the number of users in the dataset. This is needed to know how many user embeddings we have.
    :param n_items: the number of items in the dataset. This is needed to know how many item embeddings we have.
    :param emb_size: this is the dimension of the embeddings of the model. Default is 64, as suggested in the paper.
    :param dropout: this is the percentage of units that are shut down in the hidden layers of the network during
    training.
    :param seed: this is the seed for the setting of the pytorch seed. This is needed in order to have reproducible
    results.
    :param remove_double_not: this is a flag indicating whether the double negations have to be removed from the
    logical expressions or not. For example, let us assume we have the following logical expression:
    ¬a ∨ b ∨ ¬c ∨ ¬¬d ∨ ¬¬e ∨ f. If remove_double_not is set to False, then this logical expression remains unchanged,
    while if the parameter is set to True, then the logical expression becomes ¬a ∨ b ∨ ¬c ∨ d ∨ e ∨ f. In other words,
    if the parameter is set to True, ¬¬x is transformed into x. Specifically, transforming ¬¬x in x allows to avoid
    passing x through the NOT neural module two times. In fact, since ¬¬x is logically equivalent to x, it is not
    necessary to compute the double negation.
    """
    def __init__(self, n_users, n_items, emb_size=64, dropout=0.0, seed=2022, remove_double_not=False):
        super(NCR, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_size = emb_size
        self.dropout = dropout
        self.seed = seed
        # set pytorch and numpy seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)
        # initialization of user and item embeddings
        self.item_embeddings = torch.nn.Embedding(self.n_items, self.emb_size)
        self.user_embeddings = torch.nn.Embedding(self.n_users, self.emb_size)
        # this is the true anchor vector that is fixed during the training of the model (for this reason it has the
        # requires_grad parameter af False)
        self.true_vector = torch.nn.Parameter(torch.from_numpy(
            np.random.uniform(0, 0.1, size=self.emb_size).astype(np.float32)),
            requires_grad=False)  # gradient is false to disable the training of the vector
        # first layer of NOT network
        self.not_layer_1 = torch.nn.Linear(self.emb_size, self.emb_size)
        # second layer of NOT network (this network has two layers with the same number of neurons)
        self.not_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        # first layer of OR network: it takes two embeddings, so the input size is 2 * emb_size
        self.or_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        # second layer of OR network
        self.or_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        # first layer of AND network (this network is not directly used, it is used only for the logical regularizers)
        self.and_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        # second layer of AND network
        self.and_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        # first layer of encoder: it converts a pair of user-item vectors in an event vector (refer to the paper)
        self.encoder_layer_1 = torch.nn.Linear(2 * self.emb_size, self.emb_size)
        # second layer of encoder
        self.encoder_layer_2 = torch.nn.Linear(self.emb_size, self.emb_size)
        # dropout layer
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        # initialize the weights of the network
        self.init_weights()
        self.remove_double_not = remove_double_not

    def init_weights(self):
        """
        It initializes all the weights of the neural architecture as reported in the paper.
        """
        # not
        normal_init(self.not_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.not_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.not_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.not_layer_2.bias, mean=0.0, std=0.01)
        # or
        normal_init(self.or_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.or_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.or_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.or_layer_2.bias, mean=0.0, std=0.01)
        # and
        normal_init(self.and_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.and_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.and_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.and_layer_2.bias, mean=0.0, std=0.01)
        # encoder
        normal_init(self.encoder_layer_1.weight, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_1.bias, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_2.weight, mean=0.0, std=0.01)
        normal_init(self.encoder_layer_2.bias, mean=0.0, std=0.01)
        # embeddings
        normal_init(self.user_embeddings.weight, mean=0.0, std=0.01)
        normal_init(self.item_embeddings.weight, mean=0.0, std=0.01)

    def logic_not(self, vector):
        """
        This represents the NOT neural module. It takes in input an event vector and returns a new event vector that
        is the negation of the input event vector.
        :param vector: the input event vector.
        :return: the event vector that is the negation of the input event vector.
        """
        # ReLU is the activation function selected in the paper
        vector = F.relu(self.not_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.not_layer_2(vector)
        return out

    def logic_or(self, vector1, vector2, dim=1):
        """
        This represents the OR neural module. It takes in input two event vectors and returns a new event vector that is
        the logical disjunction of the two input event vectors.
        :param vector1: the first input event vector.
        :param vector2: the second input event vector.
        :param dim: the dimension for the concatenation of the two input event vectors.
        :return: the event vector that is the logical disjunction of the two input event vectors.
        """
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.or_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.or_layer_2(vector)
        return out

    def logic_and(self, vector1, vector2, dim=1):
        """
        This represents the AND neural module. It takes in input two event vectors and returns a new event vector that
        is the logical conjunction of the two input event vectors.
        :param vector1: the first input event vector.
        :param vector2: the second input event vector.
        :param dim: the dimension for the concatenation of the two input event vectors.
        :return: the event vector that is the logical conjunction of the two input event vectors.
        """
        vector = torch.cat((vector1, vector2), dim)
        vector = F.relu(self.and_layer_1(vector))
        if self.training:
            vector = self.dropout_layer(vector)
        out = self.and_layer_2(vector)
        return out

    def encoder(self, ui_vector):
        """
        This represents the encoder network of the paper. It takes in input the concatenation of two embeddings, a user
        embedding and an item embedding respectively, and it converts it into an event vector. The event vector is an
        embedding that captures the relationship between a user and an item.
        :param ui_vector: the embedding that is the concatenation of a user embedding with an item embedding.
        :return: the event vector that represents the user-item pair in input.
        """
        event_vector = F.relu(self.encoder_layer_1(ui_vector))
        if self.training:
            event_vector = self.dropout_layer(event_vector)
        event_vector = self.encoder_layer_2(event_vector)
        return event_vector

    def forward(self, batch_data):
        """
        This is the function that performs the forward phase of the neural network. In this particular network, the
        forward phase is really complex. First of all, each element in the batch represents a user-item interaction and
        comes with the following information:
            - user id: this is the id of the user.
            - item id: this is the id of the item. This is the positive item we want to predict, namely the item at the
            right side of the implication of the logical expression.
            - history: this is a numpy array that forms the history sequence of the user-item interaction. It could be
            an array of up to 5 elements. The history contains the items that have to be placed at the left side of the
            implication of the logical expression.
            - feedback of history: this is a numpy array of the same length of history. It contains only 1/0 and
            specifies which items in the history are negative items (that have to be negated with the NOT module) and
            which items are positive items. 1 means positive, while 0 means negative.
            - negative item: during training this is the negative item to build the negative logical expression for
            the pair-wise learning. During validation, instead, we have 100 negative items to build 100 negative
            expressions.

        The forward function takes as input these information for every user-item interaction in the batch, it
        constructs the positive logical expression (history -> positive item ID) and negative logical expression
        (history -> negative item ID) based on these information and finally it computes the similarity of the two
        logical expressions with the TRUE vector. An high similarity means that the logical expression is true and the
        target item could be recommended, while a low similarity means that the logical expression is false and the
        target item should not be recommended. During training, we want the logical expressions based on positive items
        to be evaluated to true, while the logical expressions based on negative items to be evaluated to false.
        Finally, this function adds all the intermediate event vectors that it obtains while constructing the
        logical expressions to a list. These intermediate event vectors are then used by the model for performing the
        logical regularization, that is needed to assure that each module learns the correct logical operator.
        :param batch_data: it contains the user-item interactions and the information to build the positive and
        negative logical expressions.
        :return positive_predictions: a torch tensor containing the similarities of the positive logical expressions
        with the TRUE vector.
        :return negative_predictions: a torch tensor containing the similarities of the negative logical expressions
        with the TRUE vector. Note that during validation we have 100 similarities for each user-item interaction in
        input since we build 100 negative logical expressions.
        :return constraints: a torch tensor of intermediate event vectors used by the model for performing logical
        regularization.
        """
        user_ids, item_ids, histories, history_feedbacks, neg_item_ids = batch_data

        # here, we select the user and item (also negative) embeddings given user, item and negative item ids
        user_embs = self.user_embeddings(user_ids)
        item_embs = self.item_embeddings(item_ids)
        neg_item_embs = self.item_embeddings(neg_item_ids)

        # here, we concatenate the user embeddings with item embeddings to get event embeddings using encoder
        # note that these are the event embeddings of the events at the right side of the implication
        right_side_events = self.encoder(torch.cat((user_embs, item_embs), dim=1))  # positive event vectors at the
        # right side of implication. These are used in the positive logical expressions.

        # in validation we have 100 negative item embeddings for each expression, while in training only 1
        # in order to make this function flexible and usable in both situations, we need to expand the user embeddings
        # in a size compatible with the negative item embeddings
        exp_user_embs = user_embs.view(user_embs.size(0), 1, user_embs.size(1))
        exp_user_embs = exp_user_embs.expand(user_embs.size(0), neg_item_embs.size(1),
                                                             user_embs.size(1))
        right_side_neg_events = self.encoder(torch.cat((exp_user_embs, neg_item_embs), dim=2))  # negative event
        # vectors at the right side of implication. These are used in the negative logical expressions.

        # now, we need the event vectors for the items at the left side of the logical expression
        # expand user embeddings to prepare for concatenating with history item embeddings
        left_side_events = user_embs.view(user_embs.size(0), 1, user_embs.size(1))
        left_side_events = left_side_events.expand(user_embs.size(0), histories.size(1), user_embs.size(1))

        # here, we get the item embeddings for the items in the histories
        history_item_embs = self.item_embeddings(histories)

        # concatenate user embeddings with history item embeddings to get left side event vectors using encoder
        left_side_events = self.encoder(torch.cat((left_side_events, history_item_embs), dim=2))

        # here, we perform the negation of the event embeddings at the left side of the expression
        left_side_neg_events = self.logic_not(left_side_events)

        # we begin to construct the constrains list containing all the intermediate event vectors used in the logical
        # regularization
        constraints = list([left_side_events])
        constraints.append(left_side_neg_events)

        # here, we take the correct (negated or not depending on the feedbacks of the users) event vectors for the
        # logic expression
        # from this instruction it begins the construction of the logical expression through the network
        # here, we expand the feedback_history tensor in order to make it compatible with left_side_events tensor, so
        # that we can perform element-wise multiplication to get the right left side event vectors
        feedback_tensor = history_feedbacks.view(history_feedbacks.size(0), history_feedbacks.size(1), 1)
        feedback_tensor = feedback_tensor.expand(history_feedbacks.size(0), history_feedbacks.size(1), self.emb_size)

        # if we do not want the double negations, we flip the feedback vector and we do not compute the NOTs in the
        # intermediate stages of the network. By doing so, we obtain a logically equivalent expression avoiding
        # the computation of double negations.
        if self.remove_double_not:
            left_side_events = (1 - feedback_tensor) * left_side_events + feedback_tensor * left_side_neg_events
        else:
            left_side_events = feedback_tensor * left_side_events + (1 - feedback_tensor) * left_side_neg_events

        #constraints = list([left_side_events])

        # now, we have the event vectors for the items in the history, we only need to build the logical expression
        # for building the logical expression we need to negate the events in the history and perform an OR operation
        # between them
        # then, we need to do (OR between negated events of the history) OR (event at the right side of implication)

        # here, we negate the events in the history only if we want to compute double negations
        if not self.remove_double_not:
            left_side_events = self.logic_not(left_side_events)
        #constraints.append(left_side_events)

        # now, we perform the logical OR between these negated left side events to build the event that is the logical
        # OR of all these events
        tmp_vector = left_side_events[:, 0]  # we take the first event of history

        shuffled_history_idx = list(range(1, histories.size(1)))  # this is needed to permute the order of the operands
        # in the OR operator at every batch
        np.random.shuffle(shuffled_history_idx)
        for i in shuffled_history_idx:
            tmp_vector = self.logic_or(tmp_vector, left_side_events[:, i])
            constraints.append(tmp_vector.view(histories.size(0), -1, self.emb_size))  # this is done to have all the
            # tensors in the constraint list of the same size
        left_side_events = tmp_vector

        constraints.append(right_side_events.view(histories.size(0), -1, self.emb_size))
        constraints.append(right_side_neg_events)  # this has already the correct shape, so it is not necessary to
        # perform a view

        # these are the event vectors of the entire original logical expressions
        expression_events = self.logic_or(left_side_events, right_side_events)
        constraints.append(expression_events.view(histories.size(0), -1, self.emb_size))

        # here, we have the result of the OR operation at the left side of the implication of the expressions in the
        # batch and we need to perform an OR between each one of these left side expressions and their corresponding
        # 1 or 100 negative right side events
        # so, for each expression we have to perform 1/100 OR operations
        # we need to reshape the results of these OR at the left side of the expressions in order to perform the OR
        # of each one with its corresponding 1/100 negative interactions
        exp_left_side_events = left_side_events.view(left_side_events.size(0), 1, left_side_events.size(1))
        exp_left_side_events = exp_left_side_events.expand(left_side_events.size(0), right_side_neg_events.size(1),
                                                           left_side_events.size(1))
        expression_neg_events = self.logic_or(exp_left_side_events, right_side_neg_events, dim=2)
        constraints.append(expression_neg_events)

        # why times 10? In order to have a sigmoid output between 0 and 1
        # these are the similarities between the positive logical expressions and the TRUE vector
        positive_predictions = F.cosine_similarity(expression_events, self.true_vector.view([1, -1])) * 10  # here the view is
        # used to transpose the true column vector in a row vector
        # these are the similarities between the negative logical expressions and the TRUE vector
        # we need to reshape the tensor containing the negative expressions in such a way to be able to compute the
        # cosine similarity of each expression with the TRUE vector
        reshaped_expression_neg_events = expression_neg_events.reshape(expression_neg_events.size(0) *
                                                        expression_neg_events.size(1), expression_neg_events.size(2))
        negative_predictions = F.cosine_similarity(reshaped_expression_neg_events, self.true_vector.view([1, -1])) * 10
        negative_predictions = negative_predictions.reshape(expression_neg_events.size(0),
                                                            expression_neg_events.size(1))

        # we convert the constraints list in tensor in order to be able compute the loss
        constraints = torch.cat(constraints, dim=1)

        # here, in order to make computation easier, we remove one dimension since it is not needed
        constraints = constraints.view(constraints.size(0) * constraints.size(1), constraints.size(2))

        return positive_predictions, negative_predictions, constraints