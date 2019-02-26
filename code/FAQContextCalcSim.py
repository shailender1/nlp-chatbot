from gensim.test.utils import datapath, get_tmpfile
from gensim.models.keyedvectors import KeyedVectors
from FAQSim import FAQSim
from gensim.scripts.glove2word2vec import glove2word2vec

import os
import numpy as np

EMBEDDING_DIM = 100
NUM_SAMPLES = 30

glove_file = datapath('C:/Users/Test/PycharmProjects/Udemy/NLP/cs224n-squad-master/data/glove.6B.100d.txt')
tmp_file = get_tmpfile("test_word2vec.txt")
_ = glove2word2vec(glove_file, tmp_file)
model = KeyedVectors.load_word2vec_format(tmp_file)

stopwords_path = "../data/stopwords_en.txt"

with open(stopwords_path, 'r') as fh:
    stopwords = fh.read().split(",")
fs = FAQSim(model,stopwords=stopwords)

## Load faq context
with open('../data/faq-sample.context', encoding="utf8") as f:
    input_answers = [current_place.rstrip() for current_place in f.readlines()]

## Load faq questions
with open('../data/faq-sample.question', encoding="utf8") as f:
    input_questions = [current_place.rstrip() for current_place in f.readlines()]

## Load faq test questions
with open('../data/faq-sample-test', encoding="utf8") as f:
    input_test = [current_place.rstrip() for current_place in f.readlines()]

target_docs = []
for a,b in zip(input_questions,input_answers):
    target_docs.append((a+ " " + b))

count = 0
for question in input_test:
    source_doc = question
    sim_scores = fs.calculate_similarity(source_doc, target_docs)
    print(question, " : ",  sim_scores)
    if(input_test.index(source_doc) == sim_scores[0]['index']):
        count +=1

print("total count: ", count)
print("Accuracy: ", (count/NUM_SAMPLES)*100)


