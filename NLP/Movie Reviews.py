from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.utils import pad_sequences
import tensorflow as tf
import numpy as np
import os

VOCAB_SIZE = 88584
MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

print(train_data[0]) # prints the encoding for each word

# Preprocessing - need to make each review the same length (padding with 0's)
train_data = pad_sequences(train_data, MAXLEN)
test_data = pad_sequences(test_data, MAXLEN)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()

# Training
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print("\n"*2,results)

# Encoding and Decoding text
word_index = imdb.get_word_index()

def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return pad_sequences([tokens], MAXLEN)[0]

reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if  num != PAD:
            text += reverse_word_index[num] + " "
    return text[:-1]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)
print(decode_integers(encoded))

# Making Prediction (higher is positive lower is more of a negative statement)
def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1,250))
    pred[0] = encoded_text
    results = model.predict(pred)
    print(results[0])

positive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great"
predict(positive_review)

negative_review = "That movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)