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
        #embedding layer
        self.G_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="We")
        #RNN cell
        self.lstm = tf.contrib.rnn.LSTMCell(self.num_units)
        #Output layer
        self.Wo = tf.Variable(tf.random_normal([self.num_units, self.vocab_size], 0.1), name="Wo")
        self.bo = tf.Variable(tf.random_normal([self.vocab_size], 0.1), name="Wo")

    def generate_examples(self, start_index = 0):
        initial_output = np.zeros((64, self.embedding_size))
        outputs = []
        m = 64
        c_state = tf.zeros([m, self.lstm.state_size[0]])
        m_state = tf.zeros([m, self.lstm.state_size[1]])
        state = (c_state, m_state)
        output = initial_output
        for t in range(self.seq_length):
            output, state = self.lstm(initial_output, state)
            z = tf.contrib.layers.fully_connected(output, self.vocab_size, activation_fn=tf.nn.relu)
            max_index = argmax(z)
            output = tf.nn.embedding_lookup(self.G_embed, [max_index])
            outputs.append(output)
        return tf.stack(outputs)



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
            z = tf.matmul(output, self.Wo) + self.bo
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

        config = tf.ConfigProto(allow_soft_placement = True)
        self.sess = tf.Session(config = config)

        with tf.device('/device:GPU:0'):
        # with tf.device('/device:CPU:0'):
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
                    print "Cost after epoch" + str(epoch) + ':' + str(minibatch_cost)

                costs.append(minibatch_cost)

            # plt.plot(np.squeeze(costs))
            # plt.ylabel('cost')
            # plt.xlabel('iterations (per tens)')
            # plt.title("Learning rate =" + str(self.learning_rate))
            # plt.show()

            correct_prediction = tf.equal(tf.argmax(self.out, 2), tf.argmax(self.labels, 2))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy)

            train_cost = 0
            train_batches = nn_tools.random_mini_batches(X_train, Y_train, self.minibatch_size, seed)
            num_train_minibatches = len(train_batches)
            for tb in train_batches:
                (mb_x, mb_y) = tb
                train_accuracy = self.sess.run(accuracy,{self.X: mb_x, self.Y: mb_y})
                train_cost += train_accuracy / num_train_minibatches
            print("Train Accuracy:", train_cost)

            test_cost = 0
            test_batches = nn_tools.random_mini_batches(X_test, Y_test, self.minibatch_size, seed)
            num_test_minibatches = len(test_batches)
            for test_mb in test_batches:
                (tmb_x, tmb_y) = test_mb
                test_accuracy = self.sess.run(accuracy,{self.X: tmb_x, self.Y: tmb_y})
                test_cost += test_accuracy / num_test_minibatches
            print("Test Accuracy:", test_cost)

            # test_accuracy = self.sess.run(accuracy,{self.X: X_test, self.Y: Y_test})
            # print("Train Accuracy:", train_accuracy)
            # print("Test Accuracy:", test_accuracy)

            # return self.report_accuracy(X_train, Y_train, X_test, Y_test)

    def adversarial_loss(self, rewards):
        probs = tf.transpose(tf.nn.softmax(self.out), perm=[1,0,2]) # (T_x, m, V) -> (m, T_x, V)
        loss = -tf.reduce_sum(
            tf.reduce_sum(
                tf.one_hot(tf.reshape(self.X, [-1]), self.vocab_size, 1.0, 0.0) * tf.log(
                    tf.clip_by_value(tf.reshape(probs, [-1, self.vocab_size]), 1e-20, 1.0)
                ), 1
            ) * tf.reshape(rewards, [-1])
        )
        return loss

    def policy_grad_update(self, rewards):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        loss = self.adversarial_loss(rewards)

        grads_and_params = optimizer.compute_gradients(loss)
        (grads, params) = np.transpose(grads_and_params).tolist()

        grads, _ = tf.clip_by_global_norm(grads, 5.0)

        return optimizer.apply_gradients(zip(grads, params))

    # def report_accuracy(self, X_train, Y_train, X_test, Y_test):


    #     return train_accuracy, test_accuracy

hparams = {
    "seq_length": 30,
    "embedding_size": 100,
    "vocab_size": 5002,
    "num_units": 300,
    "learning_rate": 1e-2,
    "num_epochs": 10,
    "minibatch_size": 500
}

G = Generator(hparams)
X = pickle.load(open('train_x.pkl', 'rb'))

X_train = X[:100000]
X_test = X[100000:101000]

Y_train = G.one_hot(X_train)
Y_test = G.one_hot(X_test)

G.train(X_train, Y_train, X_test, Y_test)
