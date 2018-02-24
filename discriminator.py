import math
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from sklearn.cross_validation import train_test_split

import nn_tools

class Discriminator:
    def __init__(self, hparams):
        """
        Assumes that hparams contains all of the following parameters
        """
        #Input Data Params
        self.seq_length = hparams["seq_length"]   # expected number of tokens per review
        self.embedding_size = hparams["embedding_size"]
        self.vocab_size = hparams["vocab_size"]
        #Network params:
        self.filter_sizes = hparams["filter_sizes"]
        self.num_filters = hparams["num_filters"]
        self.fully_connected_size = hparams["fully_connected_size"] # number of neurons in fully connected layer
        #Training params:
        self.learning_rate = hparams["learning_rate"]
        self.num_epochs = hparams["num_epochs"]
        self.minibatch_size = hparams["minibatch_size"]

        assert len(self.filter_sizes) == len(self.num_filters), "filter_sizes and num_filters must be same length"

        self.params = {}

    def train(self, X_train, Y_train, X_test, Y_test, hparams):
        ops.reset_default_graph()

        self.initialize_parameters()
        Z4 = self.forward_propagation()
        cost = self.compute_cost(Z4)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        costs = []
        seed = 1

        m = X_train.shape[0]

        with tf.Session() as sess:
            sess.run(init)

            for epoch in range(self.num_epochs):
                minibatch_cost = 0.
                num_minibatches = int(m / self.minibatch_size)
                seed += 1
                minibatches = nn_tools.random_mini_batches(X_train, Y_train, num_minibatches, seed)

                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _, temp_cost = sess.run([optimizer, cost], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                    minibatch_cost += temp_cost / num_minibatches

                if epoch % 10 == 0:
                    print("Cost after epoch", epoch, ":", minibatch_cost)
                costs.append(minibatch_cost)

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(hparams['learning_rate']))
            plt.show()

            return self.report_accuracy(Z4, X_train, Y_train, X_test, Y_test)


    def report_accuracy(self, Z4, X_train, Y_train, X_test, Y_test):
        correct_prediction = tf.equal(tf.argmax(Z4, 1), tf.argmax(self.Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy)

        train_accuracy = accuracy.eval({self.X: X_train, self.Y: Y_train})
        test_accuracy = accuracy.eval({self.X: X_test, self.Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy

    def initialize_parameters(self):
        """
        Inits placeholders & filter parameters for each provided filter shape

        *TODO: Try different initializations
        """

        self.X = tf.placeholder(tf.int32, [None, self.seq_length], name="X")
        self.Y = tf.placeholder(tf.float32, [None, 2], name="Y")

        #Embedding Layer
        W0 = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="W0")
        self.params["W0"] = W0

        #Convolutional Layers
        for i, (filter_size, num_filter) in enumerate(zip(self.filter_sizes, self.num_filters)):
            W_name = "W" + str(i+1)
            self.params[W_name] = tf.get_variable(W_name, [filter_size, self.embedding_size, 1, num_filter],
                                                    initializer=tf.initializers.truncated_normal(stddev=0.1))

    def forward_propagation(self):
        """
        Architecture:
        X -> embedding
            ----------> [f = 1] conv -> relu -> max pool
            ----------> [ ... ]
            ----------> [f =20] conv -> relu -> max pool

            ----------- All pooled outputs ------------> fully connected -> 2 neurons -> softmax output
        """
        #Embedding Layer
        W0 = self.params["W0"]
        embedded_chars = tf.nn.embedding_lookup(W0, self.X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

        #Convolutional Layers
        pooled = []
        for i in range(len(self.num_filters)):
            W = self.params["W" + str(i+1)]

            Z = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1,1,1,1], padding='VALID')
            A = tf.nn.relu(Z)
            P = tf.nn.max_pool(A, ksize=[1, self.seq_length - self.filter_sizes[i] + 1, 1, 1],
                                    strides=[1,1,1,1],
                                    padding='VALID')
            pooled.append(P)

        num_filters_total = sum(self.num_filters)
        full_pool = tf.concat(pooled, 3)
        flat_pool = tf.reshape(full_pool, [-1, num_filters_total])

        Z3 = tf.contrib.layers.fully_connected(flat_pool, self.fully_connected_size, activation_fn=tf.nn.relu)
        Z4 = tf.contrib.layers.fully_connected(Z3, 2, activation_fn=None)
        return Z4

    def compute_cost(self, Z4):
        """
        Sigmoid activation & cross entropy loss, averaged over examples
        """
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z4, labels=self.Y))
        return cost

    def basic_test(self):
        """
        Tests initialization, forward prop, cost function
        """
        with tf.Session() as sess:
            self.initialize_parameters()
            Z4 = self.forward_propagation()
            cost = self.compute_cost(Z4)

            init = tf.global_variables_initializer()
            sess.run(init)

            X_sample = np.random.randint(self.vocab_size, size=(5,self.seq_length))
            a = sess.run(Z4, {self.X: X_sample})
            print("Z4 = " + str(a))

            cost = sess.run(cost, {self.X: X_sample, self.Y: np.random.randn(5,2)})
            print("cost = " + str(cost))


hparams = {
            "seq_length": 30,
            "embedding_size": 5,
            "vocab_size": 5002,
            "filter_sizes": [1, 2],
            "num_filters": [1, 2],
            "fully_connected_size": 5,
            "learning_rate": 1e-5,
            "num_epochs": 100,
            "minibatch_size": 500
          }

D = Discriminator(hparams)
X = pickle.load(open('train_x.pkl', 'rb'))
Y = pickle.load(open('train_y.pkl', 'rb'))

X_train = X
X_test = X_train[0:100]

Y_train = Y
Y_test = Y_train[0:100]

D.train(X_train, Y_train, X_test, Y_test, hparams)

print (X_train.shape, Y_train.shape)
