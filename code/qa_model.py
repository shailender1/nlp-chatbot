
from keras.layers import Input, Embedding, LSTM,  Dense, concatenate, Concatenate, Dot, Bidirectional, RepeatVector
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
import keras.backend as K
import numpy as np
import os

# some config
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 50  # Number of epochs to train for.
LATENT_DIM = 256  # Latent dimensionality of the encoding space.
NUM_SAMPLES = 1000  # Number of samples to train on.
MAX_SEQUENCE_LENGTH = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100

# Where we will store the data
input_context= [] # answers in Q&A
input_questions = [] # questions in Q&A
target_texts = [] # answers in Q&A
target_texts_inputs = [] # answers in Q&A target offset by 1

## Load dev context
with open('../data/faq-test.context', encoding="utf8") as f:
    input_context = [current_place.rstrip() for current_place in f.readlines()]

## Load dev questions
with open('../data/faq-test.question', encoding="utf8") as f:
    input_questions = [current_place.rstrip() for current_place in f.readlines()]

## Load target answers
with open('../data/faq-test.context', encoding="utf8") as f:
    answers = [current_place.rstrip() for current_place in f.readlines()]
    for texts in answers:
        # make the target input and output
        # recall we'll be using teacher forcing
        target_text = texts + ' <eos>'
        target_text_input = '<sos> ' + texts
        target_texts.append(target_text)
        target_texts_inputs.append(target_text_input)

print("num context samples:", len(input_context))
print("num questions samples:", len(input_questions))

# tokenize the input context
tokenizer_context = Tokenizer(num_words=MAX_NUM_WORDS)
tokenizer_context.fit_on_texts(input_context)
context_input_sequences = tokenizer_context.texts_to_sequences(input_context)

# tokenize the input questions
tokenizer_questions = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_questions.fit_on_texts(input_questions)
questions_input_sequences = tokenizer_questions.texts_to_sequences(input_questions)

# tokenize the outputs
# don't filter out special characters
# otherwise <sos> and <eos> won't appear
tokenizer_outputs = Tokenizer(num_words=MAX_NUM_WORDS, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs) # inefficient, oh well
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

# determine maximum length input context sequence
max_len_context_input_sequences = max(len(s) for s in context_input_sequences)

# determine maximum length input questions sequence
max_len_questions_input_sequences = max(len(s) for s in questions_input_sequences)

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)

# get the word to index mapping for input language
word2idx_inputs = tokenizer_questions.word_index
word2idx_inputs.update(tokenizer_context.word_index)
print('Found %s unique input tokens.' % len(word2idx_inputs))

# get the word to index mapping for output response
word2idx_outputs = tokenizer_outputs.word_index
print('Found %s unique output tokens.' % len(word2idx_outputs))


# store number of output words for later
# remember to add 1 since indexing starts at 1
num_words_output = len(word2idx_outputs) + 1

# determine maximum length output sequence
max_len_target = max(len(s) for s in target_sequences)

# pad the sequences for context
context_encoder_inputs = pad_sequences(context_input_sequences, maxlen=max_len_context_input_sequences)
print("context_encoder_inputs.shape:", context_encoder_inputs.shape)

# pad the sequences for questions
questions_encoder_inputs = pad_sequences(questions_input_sequences, maxlen=max_len_questions_input_sequences)
print("questions_encoder_inputs.shape:", questions_encoder_inputs.shape)

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print("decoder_inputs.shape:", decoder_inputs.shape)

## calculate the vocab size
num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)

def get_embed_matrix(glove_dim):

    # store all the pre-trained word vectors
    print('Loading word vectors...')
    word2vec = {}
    #with open(os.path.join('C:/Users/Test/PycharmProjects/Udemy/NLP/cs224n-squad-master/data/glove.6B.%sd.txt' % EMBEDDING_DIM),encoding="utf8") as f:
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
    num_words = min(MAX_NUM_WORDS, len(word2idx_inputs) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx_inputs.items():
      if i < MAX_NUM_WORDS:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
          # words not found in embedding index will be all zeros.
          embedding_matrix[i] = embedding_vector
    return embedding_matrix
embedding_matrix = get_embed_matrix(100)

# create context embedding layer
context_embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len_context_input_sequences,
  # trainable=True
)

# create questions embedding layer
question_embedding_layer = Embedding(
  num_words,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len_questions_input_sequences,
  # trainable=True
)

# create targets, since we cannot use sparse
# categorical cross entropy when we have sequences
#   len(input_text), issue
decoder_targets_one_hot = np.zeros(
  (
    len(input_context),
    max_len_context_input_sequences,
    num_words_output
  ),
  dtype='float32'
)

def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

encoder = Bidirectional(LSTM(
    LATENT_DIM,
  return_sequences=True,
  dropout=0.5 # dropout not available on gpu
))

##### build the model #####
context_encoder_inputs_placeholder = Input(shape=(max_len_context_input_sequences,))
x1 = context_embedding_layer(context_encoder_inputs_placeholder)

print("x1 shape : ", x1.shape)

context_encoder_outputs = encoder(x1)
# encoder_outputs, h = encoder(x) #gru

questions_encoder_inputs_placeholder = Input(shape=(max_len_questions_input_sequences,))
x2 = question_embedding_layer(questions_encoder_inputs_placeholder)

questions_encoder_outputs = encoder(x2)
# encoder_outputs, h = encoder(x) #gru

print("questions_encoder_outputs: ", questions_encoder_outputs.shape)
print("context_encoder_outputs: ", context_encoder_outputs.shape)

decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)

######### Attention #########
# Attention layers need to be global because
# they will be repeated Ty times at the decoder
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(512, activation='softmax')
attn_dot = Dot(axes=-1) # to perform the weighted sum of alpha[t] * h[t]

def one_step_attention(hidden_context, hidden_question):
    context = attn_dot([hidden_context,hidden_question])
    x = attn_dense2(context)
    context2 = attn_dot([x,hidden_question])
    return context2

context = one_step_attention(context_encoder_outputs, questions_encoder_outputs)
#context2 = concatenate([context,context_encoder_outputs])
outputs = Dense(num_words_output,activation="softmax")(context)

print("output shape: ", outputs.shape)

# Create the model object
model = Model([context_encoder_inputs_placeholder, questions_encoder_inputs_placeholder], outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

r = model.fit(
  [context_encoder_inputs, questions_encoder_inputs], decoder_targets_one_hot,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  validation_split=0.2
)


decoder_lstm = LSTM(LATENT_DIM, return_state=True)
decoder_dense = Dense(num_words_output, activation='softmax')
context_last_word_concat_layer = Concatenate(axis=2)

initial_s = Input(shape=(LATENT_DIM,), name='s0')
initial_c = Input(shape=(LATENT_DIM,), name='c0')

# s, c will be re-assigned in each iteration of the loop
s = initial_s
c = initial_c

##### Make predictions #####

# map indexes into real words
# so we can view the results

context_encoder_model = Model(context_encoder_inputs_placeholder, context_encoder_outputs)
question_encoder_model = Model(questions_encoder_inputs_placeholder, questions_encoder_outputs)

context_encoder_outputs_as_input = Input(shape=(max_len_context_input_sequences, LATENT_DIM * 2,))
questions_encoder_outputs_as_input = Input(shape=(max_len_questions_input_sequences, LATENT_DIM * 2,))

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)
dense_inputs_single = Input(shape=(1,))

context = one_step_attention(context_encoder_outputs_as_input, questions_encoder_outputs_as_input)
# try combine context with last word
#decoder_lstm_input = context_last_word_concat_layer([context,decoder_inputs_single_x])

decoder_lstm_input = context_last_word_concat_layer([context, context_encoder_outputs_as_input])

# lstm and final dense
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)

# create the model object
decoder_model = Model(
  inputs=[
    decoder_inputs_single,
    context_encoder_outputs_as_input,
    questions_encoder_outputs_as_input,
    initial_s,
    initial_c
  ],
  outputs=[decoder_outputs, s, c]
)

idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_inputs.items()}

def decode_sequence(context_input_seq,questions_input_seq):
  # Encode the input as state vectors.
  context_enc_input_as_out = context_encoder_model.predict(context_input_seq)
  question_enc_input_as_out = question_encoder_model.predict(questions_input_seq)

  print("context_enc_input_as_out: ", context_enc_input_as_out.shape)
  print("question_enc_input_as_out: ", question_enc_input_as_out.shape)

  # Generate empty target sequence of length 1.
  target_seq = np.zeros((1, 1))

  # Populate the first character of target sequence with the start character.
  # NOTE: tokenizer lower-cases all words
  target_seq[0, 0] = word2idx_outputs['<sos>']

  # if we get this we break
  eos = word2idx_outputs['<eos>']

  # [s, c] will be updated in each loop iteration
  s = np.zeros((1, LATENT_DIM))
  c = np.zeros((1, LATENT_DIM))

# Create the response
  output_sentence = []
  for _ in range(max_len_target):
    o, s, c = decoder_model.predict([target_seq,context_enc_input_as_out,question_enc_input_as_out,s,c])
    # Get next word
    idx = np.argmax(o.flatten())

    # End sentence of EOS
    if eos == idx:
      break
    print("")
    #for idx in idx2word_trans:
        #print(idx, " ")
    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

  return ' '.join(output_sentence)

while True:
  # Do some test faqs
  i = np.random.choice(len(input_context))
  j = np.random.choice(len(input_questions))
  context_input_seq = context_encoder_inputs[i:i+1]
  questions_input_seq = questions_encoder_inputs[j:j+1]
  response = decode_sequence(context_input_seq,questions_input_seq)
  print('-')
  print('Question:', input_questions[i])
  print('Predicted Answer:', response)
  print('Actual Answer:', input_context[i])

  ans = input("Continue? [Y/n]")
  if ans and ans.lower().startswith('n'):
    break
