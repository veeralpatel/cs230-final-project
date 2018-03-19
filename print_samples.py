import pickle

ID_FILENAME = 'pickles/id_to_word_PUNC.pkl'
TEST_NAME = ''
OUTPUT_RANDOM = 'results/test_name_' + TEST_NAME + 'before_pretrain'
OUTPUT_LSTM = 'results/test_name_' + TEST_NAME + 'after_pretrain'
OUTPUT_GAN = 'results/test_name_' + TEST_NAME + 'after_adv'

def print_samples(samples, id_to_word):
    m, seq_length = samples.shape
    for i in range(20):
        sentence = []
        for t in range(seq_length):
            index = samples[i][t]
            sentence.append(id_to_word[index])
        print(' '.join(sentence))

index_to_word = pickle.load(open(ID_FILENAME))

random = pickle.load(open(OUTPUT_RANDOM))
lstm = pickle.load(open(OUTPUT_LSTM))
gan = pickle.load(open(OUTPUT_GAN))

print 'Random\n'
print_samples(random, index_to_word)
print '\n'

print 'LSTM\n'
print_samples(lstm, index_to_word)
print '\n' 

print 'GAN\n'
print_samples(gan, index_to_word)
print '\n'