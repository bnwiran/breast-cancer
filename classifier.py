import numpy as np
import pandas as pd
from sklearn import preprocessing


def get_data(filename):
    data = pd.read_csv(filename)
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

    scaled_data = pd.DataFrame(preprocessing.scale(data.iloc[:, 1:32]))
    scaled_data.columns = list(data.iloc[:, 1:32].columns)

    X = scaled_data.values
    Y = data['diagnosis'].values

    return X, Y


if __name__ == '__main__':
    X, Y = get_data('data.csv')

    print(len(X))
    print(len(Y))
