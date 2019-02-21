# nlp-chatbot
The goal of the project is to answer the FAQs and is based on NLP Question Answering problem. Given a text (context) and a question, the model is expected to output the answer to the question. FAQ dataset is used and the answer is a subtext of the context.

The provided baseline model consists of:
Embedding layer: The question and context inputs are processed using word embeddings vectors (GloVe)
Encoding layer: A bidirectional LSTM layer is used to encode both question and context.
Attention layer: A context to question attention distribution is computed.
Modeling layer: The attention output and the context encodings are concatenated and passed through a fully connected layer.
Output layer: The result of the modeling layer is passed to dense layer to generate output.

Steps to run the chatbot:
1. Download pre-trained embedded vectors glove.6B.100d and add to the data folder.
2. Run the file qa_model.py 

Notes:
The configuration can be changed in the file for training the model:
 - EPOCHS
 - Batch Size
 - Embedding DIM
