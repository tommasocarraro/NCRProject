import logging
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from .evaluation import ValidFunc, evaluate

__all__ = ['NCRTrainer']

logger = logging.getLogger(__name__)

class NCRTrainer(object):
    """
    This is the class that has to be used to train the NCR model. It has the following parameters:
    :param net: it is the NCR neural architecture that has to be trained.
    :param learning_rate: it is the learning rate for the Adam optimizer used by the model.
    :param l2_weight: it is the weight for the L2 regularization of the weights of the model.
    :param logic_reg_weight: it is the weight for the logical regularization performed by the model.
    """
    def __init__(self, net, learning_rate=0.001, l2_weight=1e-4, logic_reg_weight=0.01):
        self.network = net
        self.lr = learning_rate
        self.l2_weight = l2_weight
        self.reg_weight = logic_reg_weight
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.l2_weight)

    def reg_loss(self, constraints):
        """
        It computes the regularization part of the loss function.
        :param constraints: see loss_function()
        :return: the regularization loss for the batch intermediate event vectors given in input.
        """
        false_vector = self.network.logic_not(self.network.true_vector)  # we compute the representation
        # for the FALSE vector

        # here, we need to implement the logical regularizers for the logical regularization

        # here, we implement the logical regularizers for the NOT operator

        # minimizing 1 - similarity means maximizing the cosine similarity between the two event vectors in input
        # minimizing 1 + similarity means minimizing the cosine similarity between the two event vectors in input

        # here, we maximize the similarity between not not true and true
        r_not_not_true = (1 - F.cosine_similarity(
            self.network.logic_not(self.network.logic_not(self.network.true_vector)), self.network.true_vector,dim=0))

        # here, we maximize the similarity between not true and false
        #r_not_true = (1 - F.cosine_similarity(self.logic_not(self.true), false, dim=0))

        # here, we maximize the similarity between not not x and x
        r_not_not_self = \
            (1 - F.cosine_similarity(self.network.logic_not(self.network.logic_not(constraints)), constraints)).mean()

        # here, we minimize the similarity between not x and x
        r_not_self = (1 + F.cosine_similarity(self.network.logic_not(constraints), constraints)).mean()

        # here, we minimize the similarity between not not x and not x
        r_not_not_not = \
            (1 + F.cosine_similarity(self.network.logic_not(self.network.logic_not(constraints)),
                                     self.network.logic_not(constraints))).mean()

        # here, we implement the logical regularizers for the OR operator

        # here, we maximize the similarity between x OR True and True
        r_or_true = (1 - F.cosine_similarity(
            self.network.logic_or(constraints, self.network.true_vector.expand_as(constraints)),
            self.network.true_vector.expand_as(constraints))).mean()

        # here, we maximize the similarity between x OR False and x
        r_or_false = (1 - F.cosine_similarity(
            self.network.logic_or(constraints, false_vector.expand_as(constraints)), constraints)).mean()

        # here, we maximize the similarity between x OR x and x
        r_or_self = (1 - F.cosine_similarity(self.network.logic_or(constraints, constraints), constraints)).mean()

        # here, we maximize the similarity between x OR not x and True
        r_or_not_self = (1 - F.cosine_similarity(
            self.network.logic_or(constraints, self.network.logic_not(constraints)),
            self.network.true_vector.expand_as(constraints))).mean()

        # same rule as before, but we flipped operands
        r_or_not_self_inverse = (1 - F.cosine_similarity(
            self.network.logic_or(self.network.logic_not(constraints), constraints),
            self.network.true_vector.expand_as(constraints))).mean()

        # here, we implement the logical regularizers for the AND operator

        # here, we maximize the similarity between x AND True and x
        r_and_true = (1 - F.cosine_similarity(
            self.network.logic_and(constraints, self.network.true_vector.expand_as(constraints)), constraints)).mean()

        # here, we maximize the similarity between x AND False and False
        r_and_false = (1 - F.cosine_similarity(
            self.network.logic_and(constraints, false_vector.expand_as(constraints)),
            false_vector.expand_as(constraints))).mean()

        # here, we maximize the similarity between x AND x and x
        r_and_self = (1 - F.cosine_similarity(self.network.logic_and(constraints, constraints), constraints)).mean()

        # here, we maximize the similarity between x AND not x and False
        r_and_not_self = (1 - F.cosine_similarity(
            self.network.logic_and(constraints, self.network.logic_not(constraints)),
            false_vector.expand_as(constraints))).mean()

        # same rule as before, but we flipped operands
        r_and_not_self_inverse = (1 - F.cosine_similarity(
            self.network.logic_and(self.network.logic_not(constraints), constraints),
            false_vector.expand_as(constraints))).mean()

        # True/False rule

        # here, we minimize the similarity between True and False
        true_false = 1 + F.cosine_similarity(self.network.true_vector, false_vector, dim=0)

        r_loss = r_not_not_true + r_not_not_self + r_not_self + r_not_not_not + \
                 r_or_true + r_or_false + r_or_self + r_or_not_self + r_or_not_self_inverse + true_false + \
                 r_and_true + r_and_false + r_and_self + r_and_not_self + r_and_not_self_inverse

        return r_loss

    def loss_function(self, positive_preds, negative_preds, constraints):
        """
        This method computes the loss function for a single batch. It takes as inputs the predictions for positive
        and negative logical expressions and a tensor containing the intermediate event vectors obtained while building
        the logical expressions of the batch. The loss is computed as reported in the paper.
        :param positive_preds: predictions for positive logical expressions.
        :param negative_preds: predictions for negative logical expressions.
        :param constraints: tensor containing the intermediate event vectors obtained while building the logical
        expressions of the batch.
        :return the partial loss function for the given batch.
        """
        # here, we implement the recommendation pair-wise loss as in the paper
        # since we need to compute the differences between the positive predictions and negative predictions
        # we need to change the size of positive predictions in order to be of the same size of negative predictions
        # this is required because we could have more than one negative expression for each positive expression
        positive_preds = positive_preds.view(positive_preds.size(0), 1)
        positive_preds = positive_preds.expand(positive_preds.size(0), negative_preds.size(1))
        loss = -(positive_preds - negative_preds).sigmoid().log().sum()  # this is the formula in the paper

        # here, we compute the regularization loss
        r_loss = self.reg_loss(constraints)

        return loss + self.reg_weight * r_loss

    def train(self,
              train_data,
              valid_data=None,
              valid_metric=None,
              valid_func=ValidFunc(evaluate),
              num_epochs=100,
              early_stop=5,
              save_path="../saved_models/best_ncr_model.json",
              verbose=1):
        """
        This method performs the traning of the NCR model.
        :param train_data: it is a DataSampler for the training set. It loads the training data.
        :param valid_data: it is a DataSampler for the validation set. It loads the validation data.
        :param valid_metric: it is the metric that has to be computed during the validation of the model.
        :param valid_func: it is the type of evaluation used. See the evaluation mudule.
        :param num_epochs: it is the number of epochs for the training of the model.
        :param early_stop: it is the number of epochs for performing early stopping. If after early_stop epochs the
        validation metric does not increase, then the training of the model will be stopped.
        :param verbose: it is a flag indicating how many log messages have to be displayed during the training.
        """
        best_val = 0.0
        early_stop_counter = 0
        early_stop_flag = False
        if early_stop > 1:  # it means that the parameter is meaningful
            early_stop_flag = True
        try:
            for epoch in range(1, num_epochs + 1):
                self.train_epoch(epoch, train_data, verbose)
                if valid_data is not None:
                    assert valid_metric is not None, \
                                "In case of validation 'valid_metric' must be provided"
                    valid_res = valid_func(self, valid_data, valid_metric)
                    mu_val = np.mean(valid_res)
                    std_err_val = np.std(valid_res) / np.sqrt(len(valid_res))
                    logger.info('| epoch %d | %s %.3f (%.4f) |',
                                epoch, valid_metric, mu_val, std_err_val)
                    if mu_val > best_val:
                        best_val = mu_val
                        self.save_model(save_path, epoch)  # save model if an improved validation score has been
                        # obtained
                        early_stop_counter = 0  # we have reached a new validation best value, so we put the early stop
                        # counter to zero
                    else:
                        # if we did not have obtained an improved validation metric, we have to increase the early
                        # stopping counter
                        if epoch >= 20 and early_stop_flag:  # we have to train for at least 20 epochs, they said that in the paper
                            early_stop_counter += 1
                            if early_stop_counter == early_stop:
                                logger.info('Traing stopped at epoch %d due to early stopping', epoch)
                                break
        except KeyboardInterrupt:
            logger.warning('Handled KeyboardInterrupt: exiting from training early')


    def train_epoch(self, epoch, train_loader, verbose=1):
        """
        This method performs the training of a single epoch.
        :param epoch: id of the epoch.
        :param train_loader: the DataLoader that loads the training set.
        :param verbose: see train() method.
        """
        self.network.train()  # set the network in train mode
        train_loss = 0
        partial_loss = 0
        epoch_start_time = time.time()
        start_time = time.time()
        log_delay = max(10, len(train_loader) // 10**verbose)

        for batch_idx, batch_data in enumerate(train_loader):
            partial_loss += self.train_batch(batch_data)
            if (batch_idx+1) % log_delay == 0:
                elapsed = time.time() - start_time
                logger.info('| epoch %d | %d/%d batches | ms/batch %.2f | loss %.2f |',
                            epoch, (batch_idx+1), len(train_loader),
                            elapsed * 1000 / log_delay,
                            partial_loss / log_delay)
                train_loss += partial_loss
                partial_loss = 0.0
                start_time = time.time()
        total_loss = (train_loss + partial_loss) / len(train_loader)
        time_diff = time.time() - epoch_start_time
        logger.info("| epoch %d | loss %.4f | total time: %.2fs |", epoch, total_loss, time_diff)

    def train_batch(self, batch_data):
        """
        This method performs the training of a single batch.
        Parameters
        ----------
        :param batch_data: this is the batch on which we have to train on.
        :return the partial loss computed on the given batch.
        """
        self.optimizer.zero_grad()
        positive_preds, negative_preds, constraints = self.network(batch_data)
        loss = self.loss_function(positive_preds, negative_preds, constraints)
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.network.parameters(), 50)  # this has been inserted in the code provided
        self.optimizer.step()
        return loss.item()

    def predict(self, batch_data):
        """Performs the prediction on the given batch using the trained NCR network. It takes as input a batch of
        logical expressions and it returns the predictions for the positive and negative logical expressions in the
        batch. Note that during validation we have one positive expression and 100 negative expressions for each
        interaction in the batch.
        Parameters
        ----------
        batch_data : :class:`torch.Tensor`
            The input for which the prediction has to be computed.
        Returns
        -------
        :return positive_predictions: the predictions for the positive logical expressions contained in the input batch.
        There is one positive prediction for each row of the batch.
        :return negative_predictions: the predictions for the negative logical expressions contained in the inpu batch.
        There are 100 negative predictions for each row of the batch.
        """
        self.network.eval()  # we have to set the network in evaluation mode
        with torch.no_grad():
            positive_predictions, negative_predictions, _ = self.network(batch_data)
        return positive_predictions, negative_predictions

    def save_model(self, filepath, cur_epoch):
        """Save the model into the given file.
        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where to save the model.
        cur_epoch : :obj:`int`
            The current training epoch.
        """
        logger.info("Saving model checkpoint to %s...", filepath)
        torch.save({'epoch': cur_epoch,
                 'state_dict': self.network.state_dict(),
                 'optimizer': self.optimizer.state_dict()
                }, filepath)
        logger.info("Model checkpoint saved!")

    def load_model(self, filepath):
        """Load the model from the given file.
        Parameters
        ----------
        filepath : :obj:`str`
            String representing the path to the file where the model is saved.
        Returns
        -------
        :obj:`dict`
            A dictionary that summarizes the state of the model when it has been saved.
            Note: not all the information about the model are stored in the saved 'checkpoint'.
        """
        assert os.path.isfile(filepath), "The checkpoint file %s does not exist." %filepath
        logger.info("Loading model checkpoint from %s...", filepath)
        checkpoint = torch.load(filepath)
        self.network.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("Model checkpoint loaded!")
        return checkpoint

    def test(self, test_loader, metric_list=['ndcg@5', 'ndcg@10', 'hit@5', 'hit@10'], n_times=10):
        """
        This method performs the test of a trained NCR model.
        :param test_loader: this is the DataSampler that loads the test set interactions.
        :param metric_list: this is a list containing the test metrics that have to be computed.
        :param n_times: this is the number of times that the evaluation has to be computed. Since the test loader
        generates 100 random negative items for each interaction in the test set, different random generations
        could lead to different test performances. The evaluation will be computed n_times times and then each metric
        will be averaged among these n_times evaluations.
        This method will log the value of each one of the metrics (plus std error) once this procedure has finished.
        """
        metric_dict = {}
        for i in range(n_times):  # compute test metrics n_times times and take the mean since negative samples are
            # randomly generated
            evaluation_dict = evaluate(self, test_loader, metric_list)
            for metric in evaluation_dict:
                metric_mean = np.mean(evaluation_dict[metric])
                metric_std_err_val = np.std(evaluation_dict[metric]) / np.sqrt(len(evaluation_dict[metric]))
                if "mean" not in metric_dict[metric]:
                    metric_dict[metric]["mean"] = metric_mean
                    metric_dict[metric]["std"] = metric_std_err_val
                else:
                    metric_dict[metric]["mean"] += metric_mean
                    metric_dict[metric]["std"] += metric_std_err_val

        for metric in metric_dict:
            logger.info('%s: %.3f (%.4f)', metric, metric_dict[metric]["mean"] / n_times,
                        metric_dict[metric]["mean"] / n_times)