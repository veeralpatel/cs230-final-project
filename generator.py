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
        one_hot_everything = np.zeros((m, self.seq_length, self.vocab_size), dtype=np.int8)
        for m, array in enumerate(X):
            for i, number in enumerate(array):
                one_hot_everything[m][i][number] = 1
        return one_hot_everything

    def initialize_parameters(self):
        self.X = tf.placeholder(tf.int32, [None, self.seq_length], name="X")
        self.Y = tf.placeholder(tf.uint8, [None, self.seq_length, self.vocab_size], name="Y")
        self.rewards = tf.placeholder(tf.float32, [None, self.seq_length], name="rewards")
        self.sample_size = tf.placeholder(tf.int32, name="batch_size")

        #embedding layer
        self.G_embed = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0), name="We")
        #RNN cell
        self.lstm = tf.contrib.rnn.LSTMCell(self.num_units)
        #Output layer
        self.Wo = tf.Variable(tf.random_normal([self.num_units, self.vocab_size], 0.1), name="Wo")
        self.bo = tf.Variable(tf.random_normal([self.vocab_size], 0.1), name="Wo")

    def generate_examples(self, m):
        output = tf.zeros([m, self.embedding_size])
        outputs = []
        c_state = tf.zeros([m, self.lstm.state_size[0]])
        m_state = tf.zeros([m, self.lstm.state_size[1]])
        state = (c_state, m_state)

        for t in range(self.seq_length):
            output, state = self.lstm(output, state)
            z = tf.nn.softmax(self.output_layer(output))
            max_index = tf.argmax(z, axis=1)
            output = tf.nn.embedding_lookup(self.G_embed, [max_index])[0]
            outputs.append(max_index)

        examples = tf.stack(outputs)
        return tf.transpose(examples)

    def forward_propagation(self):
        embedded_words = tf.nn.embedding_lookup(self.G_embed, self.X)
        X = tf.transpose(embedded_words, perm=(1,0,2))
        m = tf.shape(X)[1]

        a0 = tf.zeros([m, self.lstm.state_size[0]]) #activation
        m0 = tf.zeros([m, self.lstm.state_size[1]]) #memory cell
        xt = tf.zeros([m, self.embedding_size])

        state = (a0, m0)
        outputs = []

        for t in range(self.seq_length):
            a, state = self.lstm(xt, state)
            z = self.output_layer(a)
            outputs.append(z)
            xt = X[t, :, :]

        return tf.stack(outputs)

    def compute_cost(self):
        self.labels = tf.transpose(self.Y, perm=(1,0,2))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out, labels=self.labels))
        return cost

    def build_graph(self):
        self.initialize_parameters()
        self.out = self.forward_propagation()
        self.cost = self.compute_cost()
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        config = tf.ConfigProto(allow_soft_placement = True)
        self.sess = tf.Session(config = config)

        self.adv_loss, self.pg_update = self.policy_grad_update()
        #self.gen_examples = self.generate_examples(self.sampling_size)
        self.gen_examples = self.rollout(0)
        self.rollouts = [self.gen_examples] + [self.rollout(t) for t in range(1, self.seq_length)]

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self, X_train, Y_train, X_test, Y_test):
        costs = []
        seed = 1
        with tf.device('/device:GPU:0'):
        # with tf.device('/device:CPU:0'):
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

    def adversarial_loss(self):
        probs = tf.transpose(tf.nn.softmax(self.out), perm=[1,0,2]) # (T_x, m, V) -> (m, T_x, V)
        loss = tf.reduce_sum( tf.one_hot(tf.reshape(self.X, [-1]), self.vocab_size, 1.0, 0.0)
                * tf.log( tf.clip_by_value(tf.reshape(probs, [-1, self.vocab_size]), 1e-20, 1.0) ), 1)
        b = tf.reshape(self.rewards, [-1])
        loss = -1 * tf.reduce_sum(loss * b)
        return loss

    def policy_grad_update(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        loss = self.adversarial_loss()

        grads_and_params = optimizer.compute_gradients(loss)
        (grads, params) = np.transpose(grads_and_params).tolist()

        grads, _ = tf.clip_by_global_norm(grads, 5.0)

        return loss, optimizer.apply_gradients(zip(grads, params))

    def rollout(self, start_t):
        embedded_words = tf.nn.embedding_lookup(self.G_embed, self.X)
        X_emb = tf.transpose(embedded_words, perm=(1,0,2))
        X = tf.transpose(self.X)
        m = tf.shape(X_emb)[1] if start_t > 0 else self.sample_size

        a0 = tf.zeros([m, self.lstm.state_size[0]]) #activation
        m0 = tf.zeros([m, self.lstm.state_size[1]]) #memory cell
        xt = tf.zeros([m, self.embedding_size])

        state = (a0, m0)
        outputs = []
        t = 0
        while t < start_t:
            _, state = self.lstm(xt, state)
            xt = X_emb[t, :, :]
            outputs.append(X[t, :])
            t += 1

        while t < self.seq_length:
            a, state = self.lstm(xt, state)
            z = tf.nn.softmax(self.output_layer(a))
            next_token = tf.cast(tf.reshape(tf.multinomial(tf.log(z), 1), [m]), tf.int32)
            xt = tf.nn.embedding_lookup(self.G_embed, next_token)
            outputs.append(next_token)
            t += 1

        out = tf.stack(outputs)
        return tf.transpose(out)

    def output_layer(self, a):
        return tf.matmul(a, self.Wo) + self.bo


    # def report_accuracy(self, X_train, Y_train, X_test, Y_test):


    #     return train_accuracy, test_accuracy

def main():
    hparams = {
        "seq_length": 30,
        "embedding_size": 5,
        "vocab_size": 5002,
        "num_units": 100,
        "learning_rate": 1e-2,
        "num_epochs": 10,
        "minibatch_size": 50
    }

    G = Generator(hparams)
    X = pickle.load(open('train_x.pkl', 'rb'))

    X_train = X[:1000]
    X_test = X[1000:1100]

    Y_train = G.one_hot(X_train)
    Y_test = G.one_hot(X_test)

    G.train(X_train, Y_train, X_test, Y_test)

    print G.generate_examples(5)


if __name__=="__main__":
    main()
