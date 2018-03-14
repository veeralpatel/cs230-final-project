import math
import pickle
import numpy as np
import tensorflow as tf
import nn_tools
from tensorflow.python.framework import ops


class LSTMCell:
    def __init__(self, emb_dim, hidden_dim):
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.initialize_parameters()

    def __call__(self, X, state):
        a_prev, c_prev = tf.unstack(state)

        # Input Gate
        i = tf.sigmoid( tf.matmul(x, self.Wxi) + tf.matmul(a_prev, self.Wai) + self.bi)
        # Forget Gate
        f = tf.sigmoid( tf.matmul(x, self.Wxf) + tf.matmul(a_prev, self.Waf) + self.bf)
        # Output Gate
        o = tf.sigmoid( tf.matmul(x, self.Wxo) + tf.matmul(a_prev, self.Wao) + self.bog)
        # New Memory Cell
        c_new = tf.nn.tanh( tf.matmul(x, self.Wxc) + tf.matmul(a_prev, self.Wac) + self.bc)
        # Final Memory cell
        c = f * c_prev + i * c_new
        # Current Hidden state
        a = o * tf.nn.tanh(c)

        return a, tf.stack([a, c])

    def initialize_parameters(self):
        self.Wxi = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Wai = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bi = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wxf = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Waf = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bf = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wxo = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Wao = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bog = tf.Variable(self.init_matrix([self.hidden_dim]))

        self.Wxc = tf.Variable(self.init_matrix([self.emb_dim, self.hidden_dim]))
        self.Wac = tf.Variable(self.init_matrix([self.hidden_dim, self.hidden_dim]))
        self.bc = tf.Variable(self.init_matrix([self.hidden_dim]))

    def init_matrix(self, shape, stddev=0.1):
        return tf.random_normal(shape, stddev=stddev)
