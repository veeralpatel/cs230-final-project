import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb')) 
id_to_word_PUNC = pickle.load(open('pickles/id_to_word_PUNC.pkl', 'rb'))

missing_count = 0
embedding_matrix = np.zeros((5001, 100))
for key, value in id_to_word_PUNC.iteritems():
    if value in model.wv.vocab:
        embedding_matrix[key] = model.wv[value]
    else:
        missing_count += 1
        embedding_matrix[key] = model.wv['<UNK>']

pickle.dump(embedding_matrix,open('embedding_matrix.pkl', 'wb'))
print missing_count
print embedding_matrix.shape
print len(model.wv.vocab)
