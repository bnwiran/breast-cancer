import numpy as np
import pandas as pd
from keras import models
from keras import layers


def get_data(filename):
    data = pd.read_csv(filename)
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

    return data


def normalize(X):
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    return X

def preprocess_data(data):
    m = data.shape[0]

    X = normalize(data.iloc[:, 1:].values)
    Y = data.iloc[:, 30].values.reshape((m, 1))

    return np.concatenate((X, Y), axis=1)


def describe_data(data):
    data_shape = data.shape
    instance_count = data_shape[0]
    features_count = data_shape[1] - 1  # Excluding the diagnosis column
    benign_count = data[data['diagnosis'] == 0].shape[0]
    malignant_count = data[data['diagnosis'] == 1].shape[0]

    return {'features_count': features_count, 'instance_count': instance_count,
            'benign_count': '{} ({:.2f}%)'.format(benign_count, (100 * benign_count / instance_count)),
            'malignant_count': '{} ({:.2f}%)'.format(malignant_count, (100 * malignant_count / instance_count))}


def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=input_shape))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


if __name__ == '__main__':
    data = get_data('data.csv')
    print(describe_data(data))

    data = preprocess_data(data)
    print(data.shape)


