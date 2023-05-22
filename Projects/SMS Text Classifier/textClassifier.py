import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# get data files
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(ROOT_DIR, 'train-data.tsv')
test_file_path = os.path.join(ROOT_DIR, 'valid-data.tsv')

dftrain = pd.read_csv(train_file_path, sep="\t", header=None, names=["type", "msg"])
dftest = pd.read_csv(test_file_path, sep="\t", header=None, names=["type", "msg"])

dftrain.dropna()
dftest.dropna()

dftrain["type"] = pd.factorize(dftrain["type"])[0]
dftest["type"] = pd.factorize(dftest["type"])[0]

train_labels =  dftrain["type"].values
test_labels =  dftest["type"].values

# train_labels = dftrain.pop("type")
# test_labels = dftest.pop("type")

trainDataset = tf.data.Dataset.from_tensor_slices((dftrain["msg"].values, train_labels))
testDataset = tf.data.Dataset.from_tensor_slices((dftest["msg"].values, test_labels))

BUFFER_SIZE = 100
BATCH_SIZE = 32
trainDataset = trainDataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
testDataset = testDataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

vec = tf.keras.layers.experimental.preprocessing.TextVectorization(output_mode='int', max_tokens=1000, output_sequence_length=1000,)
vec.adapt(trainDataset.map(lambda text, label: text))

vocab = np.array(vec.get_vocabulary())
vocab[:20]

model = tf.keras.Sequential([
    vec,
    tf.keras.layers.Embedding(
        len(vec.get_vocabulary()),
        64,
        mask_zero=True,
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

model.summary()

# Training
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])
history = model.fit(trainDataset, epochs=10, validation_data=testDataset, validation_steps=30)

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    prediction = model.predict([pred_text])
    prediction = prediction[0][0]
    return [prediction, "ham" if prediction <0.5 else "spam"]

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)

# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
    test_messages = ["how are you doing today",
                    "sale today! to stop texts call 98912460 4",
                    "i dont want to go. can we try it a different day? available sat",
                    "our new mobile video service is live. just install on your phone to start watching.",
                    "you have won £1000 cash! call to claim your prize.",
                    "i'll bring it tomorrow. don't forget the milk.",
                    "wow, is your arm alright. that happened to me one time too"
                    ]

    test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
    passed = True

    for msg, ans in zip(test_messages, test_answers):
        prediction = predict_message(msg)
        print(prediction)
        if prediction[1] != ans:
            passed = False

    if passed:
        print("You passed the challenge. Great job!")
    else:
        print("You haven't passed yet. Keep trying.")

test_predictions()