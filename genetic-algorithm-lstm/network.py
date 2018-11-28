"""Class that represents the network to be evolved."""
import random
import logging
from train import train_and_score
from math import sqrt
from numpy import array
from numpy import mean
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense

class Network():
    """Represent a network and let us operate on it.

    Currently only works for an MLP.
    """

    def __init__(self, nn_param_choices=None):
        """Initialize our network.

        Args:
            nn_param_choices (dict): Parameters for the network, includes:
                nb_neurons (list): [64, 128, 256]
                nb_layers (list): [1, 2, 3, 4]
                activation (list): ['relu', 'elu']
                optimizer (list): ['rmsprop', 'adam']
        """
        self.accuracy = 0.
        self.nn_param_choices = nn_param_choices
        self.network = {}  # (dic): represents MLP network parameters

    def create_random(self):
        """Create a random network."""
        for key in self.nn_param_choices:
            self.network[key] = random.choice(self.nn_param_choices[key])

    def create_set(self, network):
        """Set network properties.

        Args:
            network (dict): The network parameters

        """
        self.network = network

    def train(self, dataset, n_test, n_repeats=5):
        """Train the network and record the accuracy.
        
        Args:
            dataset (str): Name of dataset to use.
            
        """
        # fit and evaluate the model n times
        #scores = [train_and_score(self.network, dataset, n_test) for _ in range(n_repeats)]
        scores = train_and_score(self.network, dataset, n_test)
        # summarize score
        #result = mean(scores)
        if self.accuracy == 0.:
            self.accuracy = scores

    def print_network(self):
        """Print out a network."""
        logging.info(self.network)
        logging.info("Network RMSE: %.2f" % (self.accuracy))
