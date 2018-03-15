import tensorflow as tf
from discriminator import Discriminator
from generator import Generator
import pickle
import numpy as np

#GENERATOR HYPERPARAMETERS
EMB_DIM = 5 # embedding dimension
G_HIDDEN_UNITS = 100 # hidden state dimension of lstm cell
SEQ_LENGTH = 30 # sequence length
START_TOKEN = 0
G_EPOCH_NUM = 10 # supervise (maximum likelihood estimation) epochs
G_LEARNING_RATE = 1e-2
G_PRE_BATCH_SIZE = 50
G_SAMPLE_SIZE = 50
VOCAB_SIZE = 5002

#DISCRIMINATOR HYPERPARAMETERS
D_FILTER_SIZES = [1, 2]
D_NUM_FILTERS = [1, 2]
D_EPOCH_NUM = 100
D_BATCH_SIZE= 50
D_LEARNING_RATE = 1e-2
D_HIDDEN_UNITS = 10
D_ADV_BATCH_SIZE = 50

TOTAL_BATCH = 5

def split_data(X, Y):
	m = X.shape[0]
	permutation = list(np.random.permutation(m))
	X_p = X[permutation, :]
	Y_p = Y[permutation, :]

	X_train = X_p[:1000]
	X_test = X_p[1000:1100]

	Y_train = Y_p[:1000]
	Y_test = Y_p[1000:1100]

	return X_train, X_test, Y_train, Y_test

def get_reward(samples, rollout_num, D, G):
    rewards = []
    for i in range(rollout_num):
        for t in range(1, SEQ_LENGTH):
            gen_samples = G.rollouts[t]
            samples = G.sess.run(gen_samples, feed_dict={G.X: samples})
            ypred_for_auc = D.sess.run(tf.nn.softmax(D.Z4), feed_dict={D.X: samples})
            ypred = np.array([item[0] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[t - 1] += ypred

        # the last token reward
        feed = {D.X: samples}
        ypred_for_auc = D.sess.run(tf.nn.softmax(D.Z4), feed)
        ypred = np.array([item[0] for item in ypred_for_auc])
        if i == 0:
            rewards.append(ypred)
        else:
            rewards[SEQ_LENGTH-1] += ypred

    rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)  # batch_size x seq_length
    return rewards

def print_samples(samples, id_to_word):
    m, seq_length = samples.shape
    for i in range(m):
        sentence = []
        for t in range(seq_length):
            index = samples[t]
            sentence.append(id_to_word[index])
        print(' '.join(sentence))

def gen_pos_batch(pos_samples, batch_size):
    m = pos_samples.shape[0]
    permutation = list(np.random.permutation(m))[:batch_size]
    return pos_samples[permutation, :]

def main():
	#potential seeding here

	#################################################################################
	# 								INITIALIZATION 									#
	#################################################################################

    G_hparams = {
                    "seq_length": SEQ_LENGTH,
                    "embedding_size": EMB_DIM,
                    "vocab_size": VOCAB_SIZE,
                    "num_units": G_HIDDEN_UNITS,
                    "learning_rate": G_LEARNING_RATE,
                    "num_epochs": G_EPOCH_NUM,
                    "minibatch_size": G_PRE_BATCH_SIZE,
                    "sampling_size": G_SAMPLE_SIZE
                }

    D_hparams = {
                    "seq_length": SEQ_LENGTH,
                    "embedding_size": EMB_DIM,
                    "vocab_size": VOCAB_SIZE,
                    "filter_sizes": D_FILTER_SIZES,
                    "num_filters": D_NUM_FILTERS,
                    "fully_connected_size": D_HIDDEN_UNITS,
                    "learning_rate": D_LEARNING_RATE,
                    "num_epochs": D_EPOCH_NUM,
                    "minibatch_size": D_BATCH_SIZE
            	}

    index_to_word = pickle.load(open('id_to_word.pkl'))

    G = Generator(G_hparams)
    D = Discriminator(D_hparams)

    G_X = pickle.load(open('train_x.pkl', 'rb'))

    D_X = pickle.load(open('train_x.pkl', 'rb'))
    D_Y = pickle.load(open('train_y.pkl', 'rb'))


	#################################################################################
	# 								PRE-TRAINING 									#
	#################################################################################

    G_X_train, G_X_test, G_Y_train, G_Y_test = split_data(G_X, G_X)
    G_Y_train = G.one_hot(G_Y_train)
    G_Y_test = G.one_hot(G_Y_test)

	#TODO: Add generator samples into D_X_train (?)
    D_X_train, D_X_test, D_Y_train, D_Y_test = split_data(D_X, D_Y)
    D_pos = D_X[:277036]

    print("Started pre-training G.")
    G.build_graph()
    G.train(G_X_train, G_Y_train, G_X_test, G_Y_test)
    print("Finished training G. Started pre-training D.")
    D.train(D_X_train, D_Y_train, D_X_test, D_Y_test, report=True)
    print("Finished training D")

	#################################################################################
	# 						    ADVERARIAL-TRAINING 								#
	#################################################################################

	#Define Policy Gradient and RL loss (for generator)
    #Update D hyperparameters for adversarial training
    G_loss, G_update = G.adv_loss, G.pg_update
    D.num_epochs = 3
    D.minibatch_size = D_ADV_BATCH_SIZE

    print("Started adversarial training")

    for total_batch in range(TOTAL_BATCH):
        print "Total batch: %d" % total_batch
        # Train the generator for one step
        samples = G.sess.run(G.gen_examples)
        rewards = get_reward(samples, 16, D, G)
        _, loss = G.sess.run([G_update, G_loss], feed_dict={G.X: samples, G.rewards: rewards})

        print "Done training G. Loss: %d" % loss

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            samples = G.sess.run(G.gen_examples)
            print_samples(samples[0], index_to_word)

        # Train the discriminator
        for _ in range(5):
            samples = G.sess.run(G.gen_examples)
            pos = gen_pos_batch(D_pos, G_SAMPLE_SIZE)

            y_neg = np.tile(np.array([0,1]), (G_SAMPLE_SIZE, 1))
            y_pos = np.tile(np.array([1,0]), (G_SAMPLE_SIZE, 1))

            X_train = np.concatenate((samples, pos))
            Y_train = np.concatenate((y_neg, y_pos))

            D.train(X_train, Y_train, None, None, restart=False, report=False)

        print "Done training D."


if __name__=="__main__":
    main()
