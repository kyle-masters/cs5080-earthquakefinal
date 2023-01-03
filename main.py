import os
import make_preds, gen_series, split_data
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

# Experiment Paramaters
data_splits = 3
magnitude = 3
weeks_intervals = [(9, 14), (9, 168), (17, 14)]

# Hyperparameters to check
word_sizes = [10, 30, 50]
window_sizes = [.2, .5]
n_bins_list = [10, 14]
funcs = [lambda x: x[0], lambda x: x[1], lambda x: x[2]]  # average mag, count, max mag
func_names = ('avg', 'count', 'max')

if __name__ == '__main__':
    outputs_list = list()

    pbar = tqdm(total=data_splits*len(weeks_intervals)*len(word_sizes)*len(window_sizes)*len(n_bins_list)*len(funcs))
    for (week, interval) in weeks_intervals:
            os.makedirs(f'preds/{magnitude:1.1f}_{week}_{interval}', exist_ok=True)
            if not os.path.exists(f'data_{magnitude:1.1f}_{week}_{interval}_x.npy') or not os.path.exists(f'data_{magnitude:1.1f}_{week}_{interval}_y.pkl'):
                gen_series.generate_files(magnitude, week, interval)

            x_train, y_train, x_test, y_test = split_data.load_split(magnitude, week, interval, funcs, data_splits)

            for word_size in word_sizes:
                for window_size in window_sizes:
                    for n_bins in n_bins_list:
                        for i in range(len(funcs)):
                            to_break = False
                            aves = np.zeros((data_splits, 3*6))
                            for j in range(data_splits):
                                result = make_preds.predict(x_train[i][j], y_train[j], x_test[i][j], y_test[j], n_bins, window_size, word_size)
                                pbar.update(1)
                                # If hyper-paramaters aren't valid don't record them.
                                if not result:
                                    pbar.update(2)
                                    to_break = True
                                    break
                                aves[j] = result
                            if to_break:
                                break
                            outputs_list.append(np.append([week, interval, word_size, window_size, n_bins, func_names[i]], aves.mean(axis=0)))
    pbar.close()

    outputs_list = np.array(outputs_list)

    # Output stats on accuracy, recall, precision, F1 score, positive selection rate, negative selection rate for tested experiments to csv file.
    labels = ['accuracy', 'recall', 'precision', 'f1', 'pos-rate', 'neg-rate']
    tests = ['bop', 'boss', 'saxvsm']

    columns = ['weeks', 'intervals', 'word_size', 'window_size', 'n_bins', 'function']

    for test in tests:
        for label in labels:
            columns.append(f'{test}_{label}')

    df = pd.DataFrame(outputs_list, columns=columns)
    df.to_csv('final_stats.csv')
