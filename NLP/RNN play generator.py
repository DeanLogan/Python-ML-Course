from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os 
import numpy as np

# Dataset
path_to_file = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

# Read, then decode for py2 compat
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print("Length of text: {} characters".format(len(text)))
print("\n"+text[:250])

# Encoding and Decoding
vocab = sorted(set(text))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

def int_to_text(ints):
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

text_as_int = text_to_int(text)
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))
print("Decoded:",int_to_text(text_as_int[:13]))

# Creating Training Examples
seq_length = 100 # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int) # creates stream of characters from the text data
sequences = char_dataset.batch(seq_length+1, drop_remainder=True) # uses batch method to turn this stream of characters into batches of desired length

def split_input_target(chunk): # comments below are for the example hello as chunk
    input_text = chunk[:-1] # hell
    target_text = chunk[1:] # ello
    return input_text, target_text #hell, ello

dataset = sequences.map(split_input_target) # we use map to apply the above function to every entry

for x,y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_text(x))
    print("\nOUTPUT")
    print(int_to_text(y))

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab) # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024


BUFFER_SIZE = 10000 # Buffer size to shuffle the dataset (TF data is designed to work with possibly infinite sequences) so it doesn't attempt to shuffle the entire sequence in memory. Instead, it maintains a buffer in which it shuddles elements.

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Building the Model
