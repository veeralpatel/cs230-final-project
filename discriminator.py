import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

class Discriminator:
    def __init__(self, hparams):
        """
        Assume that hparams contains:
            - n_H0 = length of each sequence
            - n_W0 = size of word2vec (embedding size)
            - filter_sizes = array of filter sizes
            - num_filters = array of number of filters
            - learning_rate = learning rate for training
        """
        self.n_H0 = hparams["n_H0"]
        self.n_W0 = hparams["n_W0"]
        self.filter_sizes = hparams["filter_sizes"]
        self.num_filters = hparams["num_filters"]
        self.fully_connected_size = hparams["fully_connected_size"]
        self.learning_rate = hparams["learning_rate"]

        assert len(self.filter_sizes) == len(self.num_filters), "filter_sizes and num_filters must be same length"

        self.params = {}

    def train(self, X_train, Y_train):
        ops.reset_default_graph()
        self.basic_test()
        # self.initialize_parameters()
        # Z4 = self.forward_propagation(self.X, self.params)
        # cost = self.compute_cost(Z4, self.Y)
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)



    def basic_test(self):
        """
        Tests initialization, forward prop, cost function
        """
        with tf.Session() as sess:
            self.initialize_parameters()
            Z4 = self.forward_propagation(self.X, self.params)
            cost = self.compute_cost(Z4, self.Y)

            init = tf.global_variables_initializer()
            sess.run(init)
            a = sess.run(Z4, {self.X: np.random.randn(10,4,4,1)})
            cost = sess.run(cost, {self.X: np.random.randn(10,4,4,1), self.Y: np.random.randn(10,1)})
            print("Z4 = " + str(a))
            print("cost = " + str(cost))

    def initialize_parameters(self):
        """
        Inits placeholders & filter parameters for each provided filter shape

        *TODO: Try different initializations
        """
        self.X = tf.placeholder(tf.float32, [None, self.n_H0, self.n_W0, 1], name="X")
        self.Y = tf.placeholder(tf.float32, [None, 1], name="Y")

        for i, (filter_size, num_filter) in enumerate(zip(self.filter_sizes, self.num_filters)):
            W_name = "W" + str(i)
            self.params[W_name] = tf.get_variable(W_name, [filter_size, self.n_W0, 1, num_filter],
                                                    initializer=tf.initializers.truncated_normal(stddev=0.1))

    def forward_propagation(self, X, parameters):
        """
        Architecture:
            - [f = 1] conv -> relu -> max pool
            - [ ... ]
            - [f = 20] conv -> relu -> max pool

            ------- All pooled outputs -------> fully connected -> single neuron -> output

        Things to try/add:
            - dropout
            - highway (?)
            - multiple convolutional layers
            - 1-D padding
        """
        pools = []
        for i in range(len(self.num_filters)):
            W = self.params["W" + str(i)]

            Z = tf.nn.conv2d(X, W, strides=[1,1,1,1], padding='VALID')
            A = tf.nn.relu(Z)
            P = tf.nn.max_pool(A, ksize=[1, self.n_H0 - self.filter_sizes[i] + 1, 1, 1],
                                    strides=[1,1,1,1],
                                    padding='VALID')
            pools.append(P)

        num_filters_total = sum(self.num_filters)
        full_pool = tf.concat(pools, 3)
        flat_pool = tf.reshape(full_pool, [-1, num_filters_total])

        Z3 = tf.contrib.layers.fully_connected(flat_pool, self.fully_connected_size, activation_fn=tf.nn.relu)
        Z4 = tf.contrib.layers.fully_connected(Z3, 1, activation_fn=None)
        return Z4

    def compute_cost(self, Z4, Y):
        """
        Sigmoid activation & cross entropy loss, averaged over examples
        """
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Z4, labels=Y))
        return cost


hparams = { "n_H0": 4,
            "n_W0": 4,
            "filter_sizes": [1, 2],
            "num_filters": [1, 2],
            "fully_connected_size": 10,
            "learning_rate": 1e-5
          }

D = Discriminator(hparams)

D.train([], [])
