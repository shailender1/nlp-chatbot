from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Embedding, LSTM, Bidirectional
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
import keras.backend as K

import numpy as np
import os
from ManhDist import ManDist

MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
HIDDEN = 100
BATCH_SIZE = 64
EPOCHS = 1

## Load dev questions
with open('../data/faq-sample.question', encoding="utf8") as f:
    input_questions = [current_place.rstrip() for current_place in f.readlines()]

with open('../data/faq-sample-test', encoding="utf8") as f:
    input_tests = [current_place.rstrip() for current_place in f.readlines()]


tokenizer = Tokenizer(num_words=MAX_NUM_WORDS,filters='')
tokenizer.fit_on_texts(input_questions)
input_sequences = tokenizer.texts_to_sequences(input_questions)

tokenizer2 = Tokenizer(num_words=MAX_NUM_WORDS,filters='')
tokenizer2.fit_on_texts(input_tests)
input_sequences2 = tokenizer2.texts_to_sequences(input_tests)

# determine maximum length input questions sequence
max_len = max(len(s) for s in input_sequences)
max_len2 = max(len(s) for s in input_sequences2)

print("max_len: ", max_len)
print("max_len2: ", max_len2)

# get the word to index mapping for input language
word2idx_inputs = tokenizer.word_index
word2idx_inputs.update(tokenizer2.word_index)
print('Found %s unique input tokens.' % len(word2idx_inputs))

#encoder_inputs1 = pad_sequences(input_sequences, maxlen=max_len)
#encoder_inputs2 = pad_sequences(input_sequences2, maxlen=max_len2)

def get_embed_matrix(words_sim):
    # store all the pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    with open(os.path.join('../data/glove.6B.%sd.txt' % EMBEDDING_DIM),encoding="utf8") as f:
      # is just a space-separated text file in the format:
      # word vec[0] vec[1] vec[2] ...
      for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
    print('Found %s word vectors.' % len(word2vec))

    # prepare embedding matrix
    print('Filling pre-trained embeddings...')
    num_words = min(MAX_NUM_WORDS, len(words_sim) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    print("embedding_matrix initialized")
    for word, i in word2idx_inputs.items():
      if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all zeros.
          embedding_matrix[i] = embedding_vector
    print("embedding_matrix completed")
    return embedding_matrix
embedding_matrix = get_embed_matrix(word2idx_inputs)

## calculate the vocab size
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)

embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len,
  # trainable=True
)

encoder_inputs1 = pad_sequences(input_sequences, maxlen=max_len)
#encoder_inputs2 = pad_sequences(input_sequences2, maxlen=max_len2)

x = Sequential()
x.add(embedding_layer)
x.add(LSTM(HIDDEN))

model = x

# The visible layer
left_input = Input(shape=(max_len,), dtype='int32')
right_input = Input(shape=(max_len,), dtype='int32')

print("left_input: ", left_input.shape)
print("right_input: ", right_input.shape)

# Pack it all up into a Manhattan Distance model
malstm_distance = ManDist()([model(left_input), model(right_input)])

model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
model.compile(optimizer='adam', loss='mean_squared_error')

num_samples = 30
count = 0
for input_test in input_tests:
    test = []
    for i in range(num_samples):
        test.append(input_test)
    print(test)
    tokenizer3 = Tokenizer(num_words=MAX_NUM_WORDS,filters='')
    tokenizer3.fit_on_texts(test)
    input_sequences3 = tokenizer3.texts_to_sequences(test)
    encoder_inputs2 = pad_sequences(input_sequences3, maxlen=max_len)
    scores = model.predict([encoder_inputs1,encoder_inputs2])
    question_index = 0
    final_score = []
    for score in scores:
        final_score.append({
            'score' : score.astype(float),
            'index' : question_index
        })
        question_index = question_index+1
    final_score.sort(key=lambda k : k['score'])
    print(final_score)
    print("question: ", input_questions[final_score[0]['index']])
    if(input_tests.index(input_test) == final_score[0]['index']):
        count +=1
print("final count: ", count)




