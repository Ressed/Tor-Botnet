import pickle
import os
import torch
import numpy
import csv


def from_pickle(dir, name, feature=lambda x: x):
    with open(os.path.join(dir, 'X_' + name + '.pkl'), 'rb') as handle:
        flows = numpy.array(pickle.load(handle, encoding='bytes'))
        X = []
        for flow in flows:
            X.append(torch.tensor(feature(flow), dtype=torch.float))

    X = torch.nn.utils.rnn.pad_sequence(X)
    X = X.transpose(0, 1).unsqueeze(1)
    with open(os.path.join(dir, 'y_' + name + '.pkl'), 'rb') as handle:
        y = torch.tensor(numpy.array(pickle.load(handle, encoding='bytes')), dtype=torch.long)
    return X, y


def from_csv(dir, name, feature=lambda x: x):
    X = []
    y = []
    with open(os.path.join(dir, name + '.csv'), 'r') as f:
        f_reader = csv.reader(f)
        for row in f_reader:
            flow = [float(x) for x in row[1:1000]]
            # flow.extend([0] * 200)
            X.append(torch.tensor(feature(flow), dtype=torch.float))
            y.append(int(row[0]))
    X = torch.nn.utils.rnn.pad_sequence(X)
    X = X.transpose(0, 1).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.long)
    return X, y


def direction_feature(flow):
    return [(1 if x > 0 else -1) for x in flow]


def burst_feature(flow):
    res = []
    l = 1
    for i in range(1, len(flow)):
        if flow[i] * flow[i - 1] < 0:
            res.append(l)
            l = 1
        else:
            l += 1
    return res


def load_dataset(dataset_dir, data_type, feature_type):
    print("Loading dataset from", dataset_dir)
    # Point to the directory storing data

    if data_type == 'csv':
        from_func = from_csv
    elif data_type == 'pickle':
        from_func = from_pickle
    else:
        raise TypeError('Unknown data type')

    if feature_type == 'direction':
        feature_func = direction_feature
    elif feature_type == 'burst':
        feature_func = burst_feature
    else:
        feature_func = lambda x: x

    print("Data dimensions:")
    X_test, y_test = from_func(dataset_dir, 'test', feature_func)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    X_valid, y_valid = from_func(dataset_dir, 'valid', feature_func)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)

    X_train, y_train = from_func(dataset_dir, 'train', feature_func)
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
