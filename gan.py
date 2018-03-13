import tensorflow as tf
from discriminator import Discriminator
from generator import Generator
import pickle

#GENERATOR HYPERPARAMETERS
EMB_DIM = 5 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 30 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#DISCRIMINATOR HYPERPARAMETERS
dis_embedding_dim = 5
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64



def split_data(X, Y):
	m = X.shape[0]
	permutation = list(np.random.permutation(m))
	X = X[permutation, :]
	Y = Y[permutation, :].reshape((m,2))

	X_train = X[:288700]
	X_test = X[288700:]

	Y_train = Y[:288700]
	Y_test = Y[288700:]

	return X_train, X_test, Y_train, Y_test

def main():
	#potential seeding here

	#################################################################################
	# 								INITIALIZATION 									#
	#################################################################################

    G_hparams = {
        "seq_length": SEQ_LENGTH,
        "embedding_size": EMB_DIM,
        "vocab_size": 5002,
        "num_units": 100,
        "learning_rate": 1e-5,
        "num_epochs": 10
    }

	D_hparams = {
		"seq_length": SEQ_LENGTH,
		"embedding_size": EMB_DIM,
		"vocab_size": 5002,
		"filter_sizes": [1, 2],
		"num_filters": [1, 2],
		"fully_connected_size": 5,
		"learning_rate": 1e-5,
		"num_epochs": 100,
		"minibatch_size": 500
	}

	index_to_int = pickle.load(open('id_to_word.pkl'))

    G = Generator(G_hparams)
	D = Discriminator(D_hparams)

    G_X = pickle.load(open('train_x.pkl', 'rb'))

	D_X = pickle.load(open('train_x.pkl', 'rb'))
	D_Y = pickle.load(open('train_y.pkl', 'rb'))


	#################################################################################
	# 								PRE-TRAINING 									#
	#################################################################################

	G_X_train = X[:90000]
	G_X_test = X[90000:100000]
	G_Y_train = G.one_hot(G_X_train)
	G_Y_test = G.one_hot(G_X_test)

	#TODO: Add generator samples into D_X_train (?)
	D_X_train, D_X_test, D_Y_train, D_Y_test = split_data(D_X, D_Y)

    G.train(G_X_train, G_Y_train, G_X_test, G_Y_test)
	D.train(D_X_train, D_Y_train, D_X_test, D_Y_test)

	#define rollout/policy gradient (for generator)
	#alpha hyperparam

	G_update = G.policy_grad_update()


	#################################################################################
	# 						    ADVERARIAL-TRAINING 								#
	#################################################################################

	for total_batch in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = G.generate(sess)
            rewards = G.get_reward(sess, samples, 16, D)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Test
        if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            log.write(buffer)

        # Update roll-out parameters
        #rollout.update_params()

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            dis_data_loader.load_train_data(positive_file, negative_file)

            for _ in range(3):
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)
