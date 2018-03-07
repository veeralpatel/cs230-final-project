import math
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops


class Generator:

    def __init__(self, hparams):
        #Input data params
        self.seq_length = hparams["seq_length"]   # expected number of tokens per review
        self.embedding_size = hparams["embedding_size"]
        self.vocab_size = hparams["vocab_size"]
        #Model params
        self.num_units = hparams["num_units"]
        self.activation = hparams["activation"]

    def initialize_parameters(self):
        self.X = tf.placeholder(tf.int32, [self.seq_length, None, self.embedding_size], name="X")
        self.Y = tf.placeholder(tf.float32, [self.seq_length, None, self.embedding_size], name="Y")

        self.lstm = tf.contrib.rnn.LSTMCell(self.num_units)

    def forward_propagation(self):
        return tf.nn.dynamic_rnn(self.lstm, self.X)

    def train(self, X_train, Y_train, X_test, Y_test):
        self.initialize_parameters()
        outputs, state = self.forward_propagation()



    def get_reward(self, X, discriminator):
        #TODO
        reward = discriminator.predict(X)


#Input some random start state -> get next token
#   - then rollout, sample 100 different next tokens * get reward
#   -
