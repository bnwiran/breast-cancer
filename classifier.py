import numpy as np
import pandas as pd
from keras import models, optimizers, losses
from keras import layers
from time import time
import keras_metrics as km
from sklearn.model_selection import StratifiedKFold


def get_data(filename):
    data = pd.read_csv(filename)

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


def split_data(data, test_size=0.2):
    np.random.shuffle(data)
    features_count = data.shape[1]

    data_benign = data[data[:, -1] == 0]
    data_malignant = data[data[:, -1] == 1]

    benign_test_instances_count = int(test_size * data_benign.shape[0])
    malignant_test_instances_count = int(test_size * data_malignant.shape[0])

    test_data = np.concatenate((data_benign[:benign_test_instances_count], data_malignant[:malignant_test_instances_count]), axis=0)
    train_data = np.concatenate((data_benign[benign_test_instances_count:], data_malignant[malignant_test_instances_count:]), axis=0)

    np.random.shuffle(test_data)
    np.random.shuffle(train_data)

    train_data_X = train_data[:, 0:features_count - 1]
    train_data_Y = train_data[:, -1]
    test_data_X = test_data[:, 0:features_count - 1]
    test_data_Y = test_data[:, -1]

    return (train_data_X, train_data_Y), (test_data_X, test_data_Y)


def preprocess_data(data):
    data = data.drop('id', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
    m = data.shape[0]

    X = normalize(data.iloc[:, 1:].values)
    Y = data.iloc[:, 0:1].values.reshape((m, 1))

    return np.concatenate((X, Y), axis=1)


def create_model(nodes):
    model = models.Sequential()

    model.add(layers.Dense(nodes[0], activation='relu', input_shape=(30,)))
    for i in range(1, len(nodes)):
        model.add(layers.Dense(nodes[i], activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(), loss=losses.binary_crossentropy,
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

    return sensitivity, specificity, tp, fn, tn, fp


def create_and_train_model(nodes, train_data, train_labels, epochs=100, batch_size=64):
    kfold = StratifiedKFold(n_splits=10, shuffle=True)

    for train, val in kfold.split(train_data, train_labels):
        model = create_model(nodes)
        history = model.fit(train_data[train], train_labels[train], epochs=epochs, batch_size=batch_size, verbose=0)

        result = model.evaluate(train_data[val], train_labels[val], verbose=0)

    print(get_result_metrics(result))

    return model, history


def do_experiment(data, nodes=(128, 64, 64, 64)):
    data = preprocess_data(data)
    (train_data, train_labels), (test_data, test_labels) = split_data(data)

    time_start = time()
    model, history = create_and_train_model(nodes, train_data, train_labels)
    time_end = time()

    result = model.evaluate(test_data, test_labels, verbose=0)

    experiment_result = {}
    experiment_result['nodes'] = nodes
    experiment_result['training_time'] = (time_end - time_start)
    experiment_result['training_history'] = history
    experiment_result['evaluation_result'] = result

    return experiment_result


def print_experiment_result(result, file_name=None):
    nodes = result['nodes']
    training_history = get_metrics(result['training_history'])
    training_sensitivity_history = training_history[0]
    training_specificity_history = training_history[1]
    training_sensitivity = training_sensitivity_history[-1]
    training_specificity = training_specificity_history[-1]
    evaluation_result = get_result_metrics(result['evaluation_result'])
    test_sensitivity = evaluation_result[0]
    test_specificity = evaluation_result[1]
    TP = evaluation_result[2]
    FN = evaluation_result[3]
    TN = evaluation_result[4]
    FP = evaluation_result[5]
    training_time = result['training_time']

    print('Number of hidden layers: {}'.format(len(nodes)))
    print('Number of nodes in each layer: {}'.format(nodes))
    print('Training time: {:.2f} secs'.format(training_time))
    print('Test Training Metrics: Sensitivity: {:.2f}%, Specificity: {:.2f}%'.format(100 * training_sensitivity,
                                                                                     100 * training_specificity))
    print('Test Evaluation Metrics: TP: {}, FN: {}, TN: {}, FP: {}, Sensitivity: {:.2f}%, Specificity: {:.2f}%'.format(TP, FN, TN, FP, 100 * test_sensitivity,
                                                                                       100 * test_specificity))
    print('Training Sensitivity: {}'.format(training_sensitivity_history))
    print('Training Specificity: {}'.format(training_sensitivity_history))

    format_str = '\n{};{};{:.4f};{:.4f};{:.4f};{:.4f};{:.4f};{};{}'

    if file_name:
        with open(file_name, 'a') as f:
            f.write(format_str.format(len(nodes), nodes, training_time, test_sensitivity, test_specificity,
                                      training_sensitivity, training_specificity, training_sensitivity_history,
                                      training_specificity_history))


def main():
    data = get_data('data.csv')
    result = do_experiment(data)
    print_experiment_result(result, file_name='results.csv')


if __name__ == '__main__':
    np.random.seed(125)
    main()
