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

print(dftrain.head())