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