import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np 
import matplotlib.pyplot as plt

# below is used to resolve "CERTIFICATE_VERIFY_FAILED" error when trying to load the cifar10 dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() # load and split dataset

train_images, test_images = train_images / 255.0, test_images / 255.0 # Normalize pixel values to be between 0 and 1

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# uncomment to see index of an image
'''
IMG_INDEX = 1 # change this to look at other images
plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()
'''
# CNN Architecture

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) # The input shape of our data will be 32,32,3 and we will process 32 filters of size 3x3 over out input data. We will also apply the activation function relu to the output of each convolution operation.
model.add(layers.MaxPooling2D((2, 2))) # Performs the max pooling operation using 2x2 samples and stride of 2
model.add(layers.Conv2D(64, (3, 3), activation='relu')) # does the same as above but takes the feature map of the previous layer as input, also changes frequency to 64 from 32.
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.summary() # takes a look at the model so far

# Adding Dense Layers

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

# Training

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluating the Model

test_loss, test_acc = model.evaluate(test_images, test_labels, verbos=2)
print('/n'*2, test_acc)