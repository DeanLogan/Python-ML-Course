# Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from IPython.display import clear_output
# from six.moves import urllib
# import tensorflow.compat.v2.feature_column as fc 
import tensorflow as tf 
import os
import sys

# Load Dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # Training Data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # Testing Data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

'''
print("\nHead for dftrain:")
print(dftrain.head())

print("\nRow at index 0 for dftrain and y_train:")
print(dftrain.loc[0], y_train.loc[0]) # dftrain.loc[0] will locate the row at index 0, dftrain["age"] will return the column "age"

print("\nDescription for dftrain:")
print(dftrain.describe())

print("\nShape for dftrain:")
print(dftrain.shape)

print("\nHead for y_train")
print(y_train.head())
'''

# Graphs of the data uncomment plt.show() to see the graphs
dftrain.age.hist(bins=20)
#plt.show()
dftrain.sex.value_counts().plot(kind='barh')
#plt.show()
dftrain['class'].value_counts().plot(kind='barh')
#plt.show()
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
#plt.show()

CATEGOIRCAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGOIRCAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique() #gets a list of all unique values from given feature column
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

#print(feature_columns)

# Creating the input function
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function(): # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000) # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs) # split dataset into batches of 32 and repeat process for number of epochs
    return ds # return a batch of the dataset
  return input_function # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train) # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Creating the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns) # We create a linear estimtor by passing the feature columns we created earlier

# Training the model
linear_est.train(train_input_fn) # train
result = linear_est.evaluate(eval_input_fn) # get model metrics/stats by testing on testing data

print("\n\n\n\nAccuracy\n",result['accuracy']) # the result variable is simply a dict of stats about our model
print("\nResult:\n",result)

result = list(linear_est.predict(eval_input_fn))
print("\nPrediction dictionary:\n",result) # result[0] for prediction at index 0 (prediction of the persons survival at index 0)