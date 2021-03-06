import tensorflow as tf
import matplotlib.pyplot as plt
from discriminator import Discriminator
from generator import Generator
import pickle
import numpy as np
from nn_tools import random_mini_batches

ID_FILENAME = 'pickles/id_to_word_PUNC.pkl'
X_FILENAME = 'pickles/train_x_PUNC.pkl'
Y_FILENAME = 'pickles/train_y_PUNC.pkl'
GENERATOR_OUTPUT_RESULTS = 'results/test_name_normalization_more_epochs_'
EMBED_FILENAME = 'embedding_matrix.pkl'

#INITIAL DATA SPLITTING PARAMETERS
POS_CUT_OFF = 288136
ENTIRE_TRIM_SPLIT = 0.40
PRE_POST_SPLIT = 0.50

GENERATOR_TRAIN_TEST_SPLIT = 0.99
DISCRIMINATOR_TRAIN_TEST_SPLIT = 0.95

#GENERATOR HYPERPARAMETERS
EMB_DIM = 100
G_HIDDEN_UNITS = 100        #Hidden state dimension of lstm cell
SEQ_LENGTH = 30
START_TOKEN = 0
G_EPOCH_NUM = 20             #Pre-training epochs for G
G_ROLLOUT_NUM = 16
G_LEARNING_RATE = 1e-2
G_PRE_BATCH_SIZE = 50
G_PRE_SAMPLE_SIZE = 10000    #How many negative examples to pre-train D with
G_ADV_SAMPLE_SIZE = 500     #How many samples to train D with during adversarial training
G_ADV_TEST_SIZE = 10        #How many samples to print every now and then
VOCAB_SIZE = 5001
BEAM_TARGET = 1000

#DISCRIMINATOR HYPERPARAMETERS
D_FILTER_SIZES = [1, 2, 3, 5, 8, 10, 15, 20]
D_NUM_FILTERS = [10, 20, 20, 20, 20, 16, 16, 16]
D_EPOCH_NUM = 100
D_BATCH_SIZE= 100
D_LEARNING_RATE = 1e-5
D_HIDDEN_UNITS = 100
D_ADV_BATCH_SIZE = 50
D_EPOCH_NUM_ADV = 3

TOTAL_BATCH = 35

def shuffle_data(X, Y):
	m = X.shape[0]
	permutation = list(np.random.permutation(m))
	return X[permutation, :], Y[permutation, :]

def trim_whole_data(X, Y, percentage_trim):
    cutoff = int(X.shape[0]*percentage_trim)
    X = X[:cutoff]
    Y = Y[:cutoff]
    return X, Y

def split_pos_data_pretraining(X, Y, percentage_split):
    cutoff = int(X.shape[0]*percentage_split)

    X_pos_pre = X[:cutoff]
    X_pos_adv = X[cutoff:]
    Y_pos_pre = Y[:cutoff]
    Y_pos_adv = Y[cutoff:]

    return X_pos_pre, X_pos_adv, Y_pos_pre, Y_pos_adv

def split_data(X, Y, training_split):
    cutoff = int(X.shape[0]*training_split)

    X_train = X[:cutoff]
    X_test = X[cutoff:]
    Y_train = Y[:cutoff]
    Y_test = Y[cutoff:]

    return X_train, X_test, Y_train, Y_test

def print_samples(samples, id_to_word):
    m, seq_length = samples.shape
    for i in range(m):
        sentence = []
        for t in range(seq_length):
            index = samples[i][t]
            sentence.append(id_to_word[index])
        print(' '.join(sentence))

def gen_pos_batch(pos_samples, batch_size):
    m = pos_samples.shape[0]
    permutation = list(np.random.permutation(m))[:batch_size]
    return pos_samples[permutation, :]

def format_samples(pos_samples, neg_samples):
    y_neg = np.tile(np.array([0,1]), (len(neg_samples), 1))
    y_pos = np.tile(np.array([1,0]), (len(pos_samples), 1))

    X = np.concatenate((neg_samples, pos_samples))
    Y = np.concatenate((y_neg, y_pos))
    return X, Y

def get_reward(samples, rollout_num, beam, D, G):
    rewards = []
    for i in range(rollout_num):
        for t in range(1, SEQ_LENGTH):
            gen_samples = G.rollouts[t]
            samples = G.sess.run(gen_samples, feed_dict={G.X: samples, G.beam_width: beam})
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
    rewards = (rewards - np.mean(rewards, axis=1, keepdims=True)) / np.std(rewards, axis=1, keepdims=True)
    return rewards

def main():

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
                    "minibatch_size": G_PRE_BATCH_SIZE
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

    index_to_word = pickle.load(open(ID_FILENAME))
    embedding_matrix = pickle.load(open(EMBED_FILENAME))
    
    G = Generator(G_hparams, embedding_matrix)
    D = Discriminator(D_hparams)

    X = pickle.load(open(X_FILENAME, 'rb'))
    Y = pickle.load(open(Y_FILENAME, 'rb'))

    X_pos, Y_pos = trim_whole_data(X[:POS_CUT_OFF], Y[:POS_CUT_OFF], ENTIRE_TRIM_SPLIT)

    #################################################################################
    #                         GENERATOR PRE-TRAINING                                #
    #################################################################################

    # Split data into pretraining and posttraining for the generator
    X_pos_pre, X_pos_adv, Y_pos_pre, Y_pos_adv = split_pos_data_pretraining(X_pos, Y_pos, PRE_POST_SPLIT)

    # Split data into train and test for pretraining the generator
    G_X_train, G_X_test, G_Y_train, G_Y_test = split_data(X_pos_pre, X_pos_pre, GENERATOR_TRAIN_TEST_SPLIT)
    G_Y_train = G.one_hot(G_Y_train)
    G_Y_test = G.one_hot(G_Y_test)

    print("Started pre-training G.")
    G.build_graph()

    # Generate random text 
    before_pretrain_sample = G.sess.run(G.gen_examples, feed_dict={G.sample_size: G_X_train.shape[0], G.beam_width: VOCAB_SIZE})
    pickle.dump(before_pretrain_sample,open(GENERATOR_OUTPUT_RESULTS+'before_pretrain.pkl', 'wb'))

    G.train(G_X_train, G_Y_train, G_X_test, G_Y_test)
    print("Finished training G. Started pre-training D.")

    # Generate text after pretraining 
    after_pretrain_sample = G.sess.run(G.gen_examples, feed_dict={G.sample_size: G_X_train.shape[0], G.beam_width: VOCAB_SIZE})
    pickle.dump(after_pretrain_sample,open(GENERATOR_OUTPUT_RESULTS+'after_pretrain.pkl', 'wb'))

    #################################################################################
    #                         DISCRIMINATOR PRE-TRAINING                            #
    #################################################################################

    D.build_graph()
    
    for i in range(10):
        print(i)
        samples = G.sess.run(G.gen_examples, feed_dict={G.sample_size: G_PRE_SAMPLE_SIZE, G.beam_width: VOCAB_SIZE})
        D_X_train, D_Y_train = format_samples(X_pos_pre[:G_PRE_SAMPLE_SIZE], samples)
        D_X_train, D_X_test, D_Y_train, D_Y_test = split_data(D_X_train, D_Y_train, DISCRIMINATOR_TRAIN_TEST_SPLIT)
        D.train(D_X_train, D_Y_train, D_X_test, D_Y_test, report=True)
    print("Finished training D")

	#################################################################################
	# 						    ADVERSARIAL-TRAINING 								#
	#################################################################################

	#Define Policy Gradient and RL loss (for generator)
    #Update D hyperparameters for adversarial training
    G_loss, G_update = G.adv_loss, G.pg_update
    D.num_epochs = D_EPOCH_NUM_ADV
    D.minibatch_size = D_ADV_BATCH_SIZE

    print("Started adversarial training")

    D_losses = []
    G_losses = []
    beam_start = VOCAB_SIZE
    beam_rate = int((VOCAB_SIZE - BEAM_TARGET)/(TOTAL_BATCH-1))
    with tf.device('/device:GPU:0'):
    #with tf.device('/device:CPU:0'):
        for total_batch in range(TOTAL_BATCH):
            beam = beam_start
            # beam = beam_start - beam_rate*total_batch
            print "Total batch: %d" % total_batch
            # Train the generator for one step
            for g in range(1):
                samples = G.sess.run(G.gen_examples, feed_dict={G.sample_size: G_ADV_SAMPLE_SIZE, G.beam_width: beam})
                rewards = get_reward(samples, G_ROLLOUT_NUM, beam, D, G)
                _, loss = G.sess.run([G_update, G_loss], feed_dict={G.X: samples, G.rewards: rewards})
                G_losses.append(loss)
                print "Done training G. Loss: %s" % str(loss)

            # Test
            if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
                samples = G.sess.run(G.gen_examples, feed_dict={G.sample_size: G_ADV_TEST_SIZE, G.beam_width: beam})
                print_samples(samples, index_to_word)

            # Train the discriminator
            loss = 0.0
            X_train_full = []
            Y_train_full = []
            pos = gen_pos_batch(X_pos_adv, G_ADV_SAMPLE_SIZE)
            for k in range(3):
                samples = G.sess.run(G.gen_examples, feed_dict={G.sample_size: G_ADV_SAMPLE_SIZE, G.beam_width: beam})
                X_train, Y_train = format_samples(pos, samples)

                loss += 1./5 * D.train(X_train, Y_train, None, None, restart=False, report=False)
                X_train_full.append(X_train)
                Y_train_full.append(Y_train)

        test_samples = G.sess.run(G.gen_examples, feed_dict={G.sample_size: G_ADV_SAMPLE_SIZE, G.beam_width: beam})
        pos = gen_pos_batch(X_pos_adv, G_ADV_SAMPLE_SIZE)
        X_test, Y_test = format_samples(pos, test_samples)
        X_train_full = np.concatenate(X_train_full)
        Y_train_full = np.concatenate(Y_train_full)
        D_train_acc, D_test_acc = D.report_accuracy(X_train_full, Y_train_full, X_test, Y_test)

        D_losses.append(loss)
        print "Done training D. D's Loss: %s" % str(loss)

    # Generate samples after adversarial training
    after_adv_sample = G.sess.run(G.gen_examples, feed_dict={G.sample_size: G_X_train.shape[0], G.beam_width: BEAM_TARGET})
    pickle.dump(after_adv_sample,open(GENERATOR_OUTPUT_RESULTS+'after_adv.pkl', 'wb'))

    plt.subplot(1,2,1)
    plt.plot(np.squeeze(D_losses))
    pickle.dump(D_losses,open(GENERATOR_OUTPUT_RESULTS+'d_losses.pkl', 'wb'))
    plt.ylabel('Loss')
    plt.xlabel('Adversarial Epochs')
    plt.title("Discriminator")

    plt.subplot(1,2,2)
    plt.plot(np.squeeze(G_losses))
    pickle.dump(G_losses,open(GENERATOR_OUTPUT_RESULTS+'g_losses.pkl', 'wb'))
    plt.xlabel('Adversarial Epochs')
    plt.title("Generator")

    plt.savefig(GENERATOR_OUTPUT_RESULTS+'adv_learning')

if __name__=="__main__":
    main()
