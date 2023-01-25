# Imports
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import pandas as pd 

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

# Use keras (a module in tensorflow) to grab our datasets and read them into a pandas dataframe
train_path = tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path =  tf.keras.utils.get_file("iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print("Train head:\n",train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')

print("Shape of train:\n ",train.shape) # 120 entrues with 4 features

def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle and repeat if you are in training mode
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

# Feature columns describe how to use the input
my_feature_columns = []
for key in train.keys(): # loops through each of the column headers in the table (also known as the "keys")
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print("Feature Columns:\n",my_feature_columns)

# Building the model - Build a DNN (Deep Neural Network) with 2 hidden layers with 30 and 10 hidden nodes each
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns, 
    hidden_units=[30,10], # Two hidden layers of 30 & 10 nodes respectively
    n_classes=3 # The model must choose between 3 classes 
)

# Training
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True), # We include a lambda to avoid creating an inner function that can be seen in Linear Regression/titanic survival predictions
    steps=5000
)

eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))

# 2 different ways to print the model result
print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
print("\n\n\n\nModel Result Accuracy:\n",eval_result['accuracy'])

# Predicitons
def input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size) # Convert the inputs to a dataset without labels

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("\nPlease type numeric values as prompted")
for feature in features:
    valid = True
    while valid:
        val = input(feature+": ")
        if not val.isdigit(): valid= False
    predict[feature] = [float(val)]

predictions = classifier.predict(input_fn=lambda: input_fn(predict))
for pred_dict in predictions:
    print(pred_dict)
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Predicition is "{}" ({:.1f}%'.format(
        SPECIES[class_id], 100 * probability))