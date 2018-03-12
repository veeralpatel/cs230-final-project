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
        #Training params
        self.learning_rate = hparams["learning_rate"]

    def one_hot(self):
        X = pickle.load(open('train_x.pkl', 'rb'))
        one_hot_everything = []
        for array in X:
            one_hot_matrix = np.zeros((30,5002))
            for i,number in enumerate(array):
                one_hot_matrix[i][number] = 1
            one_hot_everything.append(one_hot_matrix)
        return one_hot_everything

    def initialize_parameters(self):
        self.X = tf.placeholder(tf.int32, [self.seq_length, None, self.embedding_size], name="X")
        self.Y = tf.placeholder(tf.float32, [self.seq_length, None, self.vocab_size], name="Y")

        self.lstm = tf.contrib.rnn.LSTMCell(self.num_units)

    def forward_propagation(self):
        m = tf.shape(self.X)[1]
        initial_state = tf.zeros([m, lstm.state_size])
        state = initial_state

        outputs = []

        for t in range(self.seq_length):
            output, state = self.lstm(self.X[t, : , :])
            z = tf.contrib.layers.fully_connected(output, self.vocab_size, activation_fn=tf.nn.relu)
            outputs.append(z)

        return tf.stack(outputs)

    def compute_cost(self):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.Y))
        return cost


    def rollout(self):
        #define rollout

    def train(self, X_train, Y_train, X_test, Y_test):
        self.initialize_parameters()
        self.out = self.forward_propagation()
        self.cost = self.compute_cost()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.init = global_variables_initializer()
        self.sess = tf.Session()

        for epoch in range(self.num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / self.minibatch_size)
            seed += 1
            minibatches = nn_tools.random_mini_batches(X_train, Y_train, num_minibatches, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches

            costs.append(minibatch_cost)

        return costs

    def update(self, initial_state, ):
        numpy_state = initial_state.eval()

        total_loss = 0.0
        for current_batch_of_words in words_in_dataset:
            numpy_state, current_loss = session.run([final_state, loss],
            # Initialize the LSTM state from the previous iteration.
            feed_dict={initial_state: numpy_state, words: current_batch_of_words})
    
        total_loss += current_loss

    def get_reward(self, X, discriminator):
        #TODO
        reward = discriminator.predict(X)
