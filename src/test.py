import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from random import seed
from random import randrange
from csv import reader
from math import sqrt

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

data = pd.read_csv('datasets/mushrooms.csv')
labelencoder = LabelEncoder()
for col in data.columns:
    data[col] = labelencoder.fit_transform(data[col])

data.to_csv('datasets/mushrooms_custom.csv')
filename = 'datasets/mushrooms_custom.csv'
dataset = load_csv(filename)
del dataset[0]
print(dataset)

# print(df.head())