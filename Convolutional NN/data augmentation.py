import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np 
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img, img_to_array # used 


# below is used to resolve "CERTIFICATE_VERIFY_FAILED" error when trying to load the cifar10 dataset
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data() # load and split dataset

train_images, test_images = train_images / 255.0, test_images / 255.0 # Normalize pixel values to be between 0 and 1

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Data Augmentation

# creates a data generator object that transforms images
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# pick an image to transform
test_img = train_images[14]
img = tf.keras.utils.img_to_array(test_img) # convert image to numpy array
img = img.reshape((1,) + img.shape) # reshape image

i = 0
for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'): # this loop runs forever until we break, saving images to current directory
    plt.figure(i)
    plot = plt.imshow(tf.keras.utils.img_to_array(batch[0]))
    i += 1
    if i > 4:
        break

plt.show()