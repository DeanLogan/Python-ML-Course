import tensorflow as tf
from tensorflow import keras
import numpy as np 
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # split inot testing and training datasets

print(f"Shape of the train_images {train_images.shape}")
print(f"Looking at 1 pixel {train_images[0,23,23]}") # lets have a look at one pixel
print(f"The first 10 training labels {train_labels[:10]}") # lets have a look at the first 10 training labels

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#uncommnet below code to see an image from the training data set
'''
plt.figure
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()
'''

# Data PreProcessing (want the range of the data values to be between 0-1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Building the Model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), # input layer (1)
    keras.layers.Dense(128, activation='relu'), # hidden layer (2)
    keras.layers.Dense(10, activation='softmax') # output layer (3)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=1) # accuracy on training data (a higher epochs can cause over fitting however if it's too low then it may not be trained enough)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1) # testing model

print('Test accuracy: ',test_acc)
print('Test loss: ', test_loss)

# Making Predictions

predictions = model.predict(test_images) # this will return a 2d array of the prediction for all the images within test_images the inner array will contain how confident the model is for each class type.

print(np.argmax(predictions[0])) # returns the index of the class (from class_names) that it is most confident in e.g returns 9 for Ankle Boot

print("\n"*5)

# Verifying Predictions

COLOUR = 'black'
plt.rcParams['text.color'] = COLOUR
plt.rcParams['axes.labelcolor'] = COLOUR

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Excpected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()


def predictionFunction():
    while True:
        num = input("Pick a number (enter quit to stop): ")
        if num.isdigit():
            num = int(num)
            if 0<= num <= 1000:
                image = test_images[num]
                label = test_labels[num]
                predict(model, image, label)
        elif num == "quit":
            print("Exiting code...\n")
            break
        else:
            print("Enter quit to stop or a number for the model between 0 and 1000")

if __name__ == "__main__":
    predictionFunction()