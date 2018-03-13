import math
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import nn_tools
from tensorflow.python.framework import ops


class Generator:

    def __init__(self, hparams):
        #Input data params
        self.seq_length = hparams["seq_length"]   # expected number of tokens per review
        self.embedding_size = hparams["embedding_size"]
        self.vocab_size = hparams["vocab_size"]
        #Model params
        self.num_units = hparams["num_units"]
        #Training params
        self.learning_rate = hparams["learning_rate"]
        self.num_epochs = hparams["num_epochs"]
        self.minibatch_size = hparams["minibatch_size"]

    def one_hot(self, X):
        m = X.shape[0]
        one_hot_everything = np.zeros((m, 30, 5002), dtype=np.int8)
        for m, array in enumerate(X):
            for i, number in enumerate(array):
                one_hot_everything[m][i][number] = 1
        return one_hot_everything

    def initialize_parameters(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_length], name="X")
        self.Y = tf.placeholder(tf.uint8, [None, self.seq_length, self.vocab_size], name="Y")
        self.G_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="We")
        self.lstm = tf.contrib.rnn.LSTMCell(self.num_units)

    def forward_propagation(self):
        embedded_words = tf.nn.embedding_lookup(self.G_embed, self.X)
        X = tf.transpose(embedded_words, perm=(1,0,2))
        m = tf.shape(X)[1]

        c_state = tf.zeros([m, self.lstm.state_size[0]])
        m_state = tf.zeros([m, self.lstm.state_size[1]])
        state = (c_state, m_state)
        outputs = []

        for t in range(self.seq_length):
            output, state = self.lstm(X[t, :, :], state)
            z = tf.contrib.layers.fully_connected(output, self.vocab_size, activation_fn=tf.nn.relu)
            outputs.append(z)

        return tf.stack(outputs)

    def compute_cost(self):
        self.labels = tf.transpose(self.Y, perm=(1,0,2))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels))
        return cost

    def rollout(self):
        pass

    def train(self, X_train, Y_train, X_test, Y_test):
        self.initialize_parameters()
        self.out = self.forward_propagation()
        self.cost = self.compute_cost()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()

        with tf.device('/device:GPU:0'):


            costs = []
            seed = 1
            self.sess.run(self.init)

            print("Starting training")
            for epoch in range(self.num_epochs):
                minibatch_cost = 0.
                seed += 1
                minibatches = nn_tools.random_mini_batches(X_train, Y_train, self.minibatch_size, seed)
                num_minibatches = len(minibatches)
                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _, temp_cost = self.sess.run([self.optimizer, self.cost], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})
                    minibatch_cost += temp_cost / num_minibatches

                if epoch % 2 == 0:
                    print("Cost after epoch", epoch, minibatch_cost)

                costs.append(minibatch_cost)

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()
            return self.report_accuracy(X_train, Y_train, X_test, Y_test)

    def update(self, initial_state):
        pass
        numpy_state = initial_state.eval()

        total_loss = 0.0
        for current_batch_of_words in words_in_dataset:
            numpy_state, current_loss = session.run([final_state, loss],
            # Initialize the LSTM state from the previous iteration.
            feed_dict={initial_state: numpy_state, words: current_batch_of_words})

        total_loss += current_loss

    def get_reward(self, X, discriminator):
        #TODO
        #X should be of shape (N, 30, 5002)
        # go through each N, sum prediction from discriminator
        reward = discriminator.predict(X)

    def report_accuracy(self, X_train, Y_train, X_test, Y_test):
        correct_prediction = tf.equal(tf.argmax(self.out, 2), tf.argmax(self.labels, 2))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy)

        train_accuracy = self.sess.run(accuracy,{self.X: X_train, self.Y: Y_train})
        test_accuracy = self.sess.run(accuracy,{self.X: X_test, self.Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy

hparams = {
    "seq_length": 30,
    "embedding_size": 5,
    "vocab_size": 5002,
    "num_units": 100,
    "learning_rate": 1e-5,
    "num_epochs": 10,
    "minibatch_size": 50
}

G = Generator(hparams)
X = pickle.load(open('train_x.pkl', 'rb'))

X_train = X[:500]
X_test = X[500:1000]

Y = G.one_hot(X_train)

Y_train = Y[:500]
Y_test = Y[500:1000]

G.train(X_train, Y_train, X_test, Y_test)
