import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # split inot testing and training datasets

print(f"Shape of the train_images {train_images.shape}")
print(f"Looking at 1 pixel {train_images[0,23,23]}") # lets have a look at one pixel
print(f"The first 10 training labels{train_labels[:10]}") # lets have a look at the first 10 training labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Bag', 'Ankle boot']

plt.figure
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# Data PreProcessing (want the range of the data values to be between 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the Model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,29)), # input layer (1)
    keras.layers.Dense(128, activation='relu'), # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])