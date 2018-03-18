
# coding: utf-8

# In[21]:


import tensorflow as tf
import gensim


# In[22]:


reviews = []

with open('train_set_1.txt') as train:
    for line in train.readlines():
        reviews.append(line)


# In[13]:


import re

trimmed_reviews = []
sentences = []
i = 0
print len(reviews)
for review in reviews:
    review = review.lower()
    review = re.sub('([.,!?()])', r' \1 ', review)
    review = re.sub('\s{2,}', ' ', review)
    review = review.split(' ')
    if len(review) > 30:
        trimmed_reviews.append(review[:30])
        sentences.append(review[:30])
    i +=1


# In[14]:


count = 0
for review in trimmed_reviews:
    for word in review:
        count +=1
count


# In[15]:


vocab = {}
for review in trimmed_reviews:
    for word in review:
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1


# In[ ]:


import operator

vocab_sorted = dict(sorted(vocab.iteritems(), key=operator.itemgetter(1), reverse=True)[:25000])
vocab_sorted


# In[17]:


vocab_ids = {}
id = 0
for key in vocab_sorted.keys():
    vocab_ids[key] = id
    id += 1
print id


# In[18]:


id_to_word = {v: k for k, v in vocab_ids.iteritems()}
len(id_to_word)


# In[19]:


id_to_word[25000] = '<UNK>'
id_to_word[25000]


# In[20]:


for i, review in enumerate(trimmed_reviews):
    if i % 10000 == 0:
        print(i)
    for j, word in enumerate(review):
        word = word.lower()
        if word in vocab_sorted.keys():
            trimmed_reviews[i][j] = vocab_ids[word]
        else:
            trimmed_reviews[i][j] = 25000
trimmed_reviews


# In[ ]:


import numpy as np
x = np.asarray(trimmed_reviews)
import pickle
pickle.dump(x,open( "embed_x.pkl", "wb"))
pickle.dump(id_to_word,open( "id_to_word_embed.pkl", "wb"))


# In[ ]:


X = pickle.load(open('embed_x.pkl', 'rb'))
id_to_word = pickle.load(open("id_to_word_embed.pkl", "rb"))


# In[ ]:


model = gensim.models.word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)


# In[ ]:


keys= model.wv.vocab.keys()
print(len(keys))
# for i, sentence in enumerate(sentences):
#     if i % 1000 == 0:
#         print(i)
#     for j, word in enumerate(sentence):
#         if word not in keys:
#             sentences[i] = sentence[0:j] + ['<UNK>'] + sentence[j+1:]
            


# In[92]:





# In[93]:


model = gensim.models.word2vec.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
pickle.dump(model, open('model.pkl', 'wb'))
#vocab = model.wv.vocab.keys()
#model.save('word2vec_trained.pkl')


# In[96]:


model =  pickle.load(open('model.pkl', 'rb'))#gensim.utils.SaveLoad.load('word2vec_trained')

