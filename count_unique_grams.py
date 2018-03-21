import pickle

POS_CUT_OFF = 288136
X_FILENAME = 'pickles/train_x_PUNC.pkl'

X = pickle.load(open(X_FILENAME, 'rb'))

def trim_training_data(X, percentage_trim):
    cutoff = int(X.shape[0]*percentage_trim)
    X = X[:cutoff]
    return X

X_pos = trim_training_data(X[:POS_CUT_OFF], 1.0)  # make sure this percentage matches whatever we trained the models on
X_pos = X_pos[:114101]

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])

data = X_pos.reshape(-1)

unique_bigrams_data = set(find_ngrams(data,2))
unique_trigrams_data = set(find_ngrams(data,3))
unique_quadgrams_data = set(find_ngrams(data,4))

generated_text = pickle.load(open('results/test_name_normalization_more_epochs_after_adv.pkl', 'rb'))
generated_text = generated_text.reshape(-1)

unique_bigrams_model = set(find_ngrams(generated_text,2))
unique_trigrams_model = set(find_ngrams(generated_text,3))
unique_quadgrams_model = set(find_ngrams(generated_text,4))

intersection_bigram = set.intersection(unique_bigrams_data,unique_bigrams_model)
intersection_trigram = set.intersection(unique_trigrams_data,unique_trigrams_model)
intersection_quadgram = set.intersection(unique_quadgrams_data,unique_quadgrams_model)

print 'Brigram'
print len(intersection_bigram)/(1.0*len(unique_bigrams_data))

print 'Trigram'
print len(intersection_trigram)/(1.0*len(unique_trigrams_data))

print 'Quadgram'
print len(intersection_quadgram)/(1.0*len(unique_quadgrams_data))





