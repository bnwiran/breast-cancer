import numpy as np
import pandas as pd
from keras import models, optimizers, losses, metrics
from keras import layers
from time import time
import keras_metrics as km


def get_data(filename):
    data = pd.read_csv(filename)
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

    return data


def describe_data(data):
    data_shape = data.shape
    instance_count = data_shape[0]
    features_count = data_shape[1] - 1  # Excluding the diagnosis column
    benign_count = data[data['diagnosis'] == 0].shape[0]
    malignant_count = data[data['diagnosis'] == 1].shape[0]

    return {'features_count': features_count, 'instance_count': instance_count,
            'benign_count': '{} ({:.2f}%)'.format(benign_count, (100 * benign_count / instance_count)),
            'malignant_count': '{} ({:.2f}%)'.format(malignant_count, (100 * malignant_count / instance_count))}


def normalize(X):
    X -= X.mean(axis=0)
    X /= X.std(axis=0)

    return X


def split_data(data, test_size=0.1):
    m = data.shape[0]
    features_count = data.shape[1]
    test_instances_count = int(test_size * m)
    test_data = data[:test_instances_count]
    train_data = data[test_instances_count:]

    train_data_X = train_data[:, 0:features_count - 1]
    train_data_Y = train_data[:, features_count - 1:features_count]
    test_data_X = test_data[:, 0:features_count - 1]
    test_data_Y = test_data[:, features_count - 1:features_count]

    return (train_data_X, train_data_Y), (test_data_X, test_data_Y)


def preprocess_data(data):
    m = data.shape[0]

    X = normalize(data.iloc[:, 1:].values)
    Y = data.iloc[:, 0:1].values.reshape((m, 1))

    return np.concatenate((X, Y), axis=1)


def create_model(nodes=(2048, 2048)):
    model = models.Sequential()

    model.add(layers.Dense(nodes[0], activation='relu', input_shape=(30,)))
    for i in range(1, len(nodes)):
        model.add(layers.Dense(nodes[i], activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss=losses.binary_crossentropy,
                  metrics=[km.binary_true_positive(), km.binary_false_negative(),
                           km.binary_true_negative(), km.binary_false_positive()])

    return model


def get_metrics(history):
    tp = history.history['true_positive']
    fn = history.history['false_negative']
    tn = history.history['true_negative']
    fp = history.history['false_positive']

    sensitivity = [tp[i] / (tp[i] + fn[i]) for i in range(len(tp))]
    specificity = [tn[i] / (tn[i] + fp[i]) for i in range(len(tp))]

    return sensitivity, specificity


def get_result_metrics(result):
    tp = result[1]
    fn = result[2]
    tn = result[3]
    fp = result[4]

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return sensitivity, specificity


def main():
    data = get_data('data.csv')
    print(describe_data(data))

    data = preprocess_data(data)
    (train_data, train_labels), (test_data, test_labels) = split_data(data)

    model = create_model()

    time_start = time()
    history = model.fit(train_data, train_labels, epochs=20, batch_size=128, verbose=0)
    time_end = time()

    print(get_metrics(history))

    result = model.evaluate(test_data, test_labels)
    sensitivity, specificity = get_result_metrics(result)
    print(sensitivity)
    print(specificity)
    print(time_end - time_start)


if __name__ == '__main__':
    main()
