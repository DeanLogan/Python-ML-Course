import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # split inot testing and training datasets

print(f"Shape of the train_images {train_images.shape}")

print(f"Looking at 1 pixel {train_images[0,23,23]}") # lets have a look at one pixel

print(f"The first 10 training labels{train_labels[:10]}") # lets have a look at the first 10 training labels
