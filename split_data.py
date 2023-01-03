from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pickle


# Split data with stratified shuffle along however many splits
def split_data(data_x, data_y, data_splits):
        sss = StratifiedShuffleSplit(n_splits=data_splits, test_size=0.3)

        X_train, X_test = list(), list()
        y_train, y_test = list(), list()

        for train_index, test_index in sss.split(data_x, data_y):
            X_train.append(data_x[train_index])
            X_test.append(data_x[test_index])
            y_train.append(data_y[train_index])
            y_test.append(data_y[test_index])

        return X_train, y_train, X_test, y_test


# Load the data files and create splits with functions (average magnitude, event count, maximum magnitude)
def load_split(magnitude, weeks, intervals, funcs, data_splits):
    data_x = np.load(f'data_{magnitude:1.1f}_{weeks}_{intervals}_x.npy', allow_pickle=True)
    with open(f'data_{magnitude:1.1f}_{weeks}_{intervals}_y.pkl', 'rb') as f:
        data_y = pickle.load(f)

    data_y = np.array([1 if data_y[i] else 0 for i in range(len(data_y))])
    x_train, y_trains, x_test, y_tests = split_data(data_x, data_y, data_splits)

    # Make copies of the np arrays so that functions can be independently applied
    x_trains = [[x_train[n].copy() for n in range(data_splits)] for m in range(len(funcs))]
    x_tests = [[x_test[n].copy() for n in range(data_splits)] for m in range(len(funcs))]

    # For each split, perform each function
    for m in range(len(funcs)):
        for n in range(data_splits):
            for i in range(x_trains[m][n].shape[0]):
                for j in range(x_trains[m][n].shape[1]):
                    x_trains[m][n][i, j] = funcs[m](x_trains[m][n][i, j])
            for i in range(x_tests[m][n].shape[0]):
                for j in range(x_tests[m][n].shape[1]):
                    x_tests[m][n][i, j] = funcs[m](x_tests[m][n][i, j])

    return x_trains, y_trains, x_tests, y_tests
