import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# get data files
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_DIR = os.path.join(ROOT_DIR, 'insurance.csv')
dataset = pd.read_csv(FILE_DIR)
print(dataset.head())

# change any strings to ints
dataset["sex"] = pd.factorize(dataset["sex"])[0]
dataset["region"] = pd.factorize(dataset["region"])[0]
dataset["smoker"] = pd.factorize(dataset["smoker"])[0]

dftest = dataset.sample(frac=0.2) # selects 20% of the df randomly
#dftest = dataset.tail(round(len(dataset)*0.2)) # uncomment this line and comment the line above to use the last 20% of the dataset as the training data
dftrain = dataset[~dataset.isin(dftest)].dropna() # removes all the records within dftrain from dataset and stores this as dftest

# remove expenses fromand store
train_labels = dftrain.pop('expenses')
test_labels = dftest.pop('expenses')

print(dftrain.head())

# normalize the data and 
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(np.array(dftrain))

model = tf.keras.Sequential([
    normalizer,
    tf.keras.layers.Dense(26),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1),
])

model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.05),
    loss='mae',
    metrics=['mae', 'mse']
)
model.build()
model.summary()

history = model.fit(
    dftrain,
    train_labels,
    epochs=225,
    validation_split=0.4,
    verbose=0, # Suppress logging
)

print(history)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = model.evaluate(dftest, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
    print("You passed the challenge. Great job!")
else:
    print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = model.predict(dftest).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims,lims)