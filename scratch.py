import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# THIS FILE HAS CHANGED SO MANY TIMES. It was used for plot generation, I'm not sure if all the plots that were generated
# are represented in this file because I did a lot of that in a command line interface.

df = pd.read_csv('final_stats.csv')

bop_values = list()
boss_values = list()
sax_values = list()
names = list()
for week in df['weeks'].unique():
    for interval in df['intervals'].unique():
        bop_values.append(df[(df.weeks == week) & (df.intervals == interval)]['bop_f1'].to_numpy())
        boss_values.append(df[(df.weeks == week) & (df.intervals == interval)]['boss_f1'].to_numpy())
        sax_values.append(df[(df.weeks == week) & (df.intervals == interval)]['saxvsm_f1'].to_numpy())
        if bop_values[-1].shape[0] > 0:
            names.append(f'Weeks: {week}\n Intervals: {interval}')
        else:
            bop_values = bop_values[:-1]
            boss_values = boss_values[:-1]
            sax_values = sax_values[:-1]

def make_plot(values, name):
    plt.bar(list(range(len(values))), [np.max(value) for value in values], tick_label=names, color='#007668')
    plt.errorbar(list(range(len(values))), [np.mean(value) for value in values],
                 [[np.mean(value) - np.min(value) for value in values],
                  [np.max(value) - np.mean(value) for value in values]],
                 fmt='o', color='black', capsize=10)
    plt.ylabel('F1 Score')
    plt.ylim(.1, .3)
    plt.title(name)
    plt.show()

make_plot(bop_values, 'BOP: F1 Score by weeks/intervals')
make_plot(boss_values, 'BOSS: F1 Score by weeks/intervals')
make_plot(sax_values, 'SAXVSM: F1 Score by weeks/intervals')

bop_values = []
boss_values = []
sax_values = []
bop_names = []
boss_names = []
sax_names = []
for category in ['word_size', 'window_size', 'n_bins', 'function']:
    bop_values.append([])
    boss_values.append([])
    sax_values.append([])
    bop_names.append([])
    boss_names.append([])
    sax_names.append([])
    for value in df[category].unique():
        curr_values = df[(df.weeks == 9) & (df.intervals == 14) & (df[category] == value)]['bop_f1'].to_numpy()
        if curr_values.shape[0] > 0:
            bop_values[-1].append(curr_values)
            bop_names[-1].append(value)
        curr_values = df[(df.weeks == 9) & (df.intervals == 14) & (df[category] == value)]['boss_f1'].to_numpy()
        if curr_values.shape[0] > 0:
            boss_values[-1].append(curr_values)
            boss_names[-1].append(value)
        curr_values = df[(df.weeks == 9) & (df.intervals == 14) & (df[category] == value)]['saxvsm_f1'].to_numpy()
        if curr_values.shape[0] > 0:
            sax_values[-1].append(curr_values)
            sax_names[-1].append(value)


def make_plot_values(values, test, category, names):
    plt.bar(list(range(len(values))), [np.max(value) for value in values], tick_label=names, color='#007668')
    plt.errorbar(list(range(len(values))), [np.mean(value) for value in values],
                 [[np.mean(value) - np.min(value) for value in values],
                  [np.max(value) - np.mean(value) for value in values]],
                 fmt='o', color='black', capsize=10)
    plt.ylabel('F1 Score')
    plt.ylim(.1, .3)
    plt.title(f'{test}: F1 Score by {category}')
    plt.show()


tests = [bop_values, boss_values, sax_values]
tests1 = [bop_names, boss_names, sax_names]
test_names = ['BOP', 'BOSS', 'SAXVSM']
categories = ['Word Size', 'Window Size', 'n_bins', 'Data Point']

for i in range(len(tests)):
    for j in range(len(categories)):
        make_plot_values(tests[i][j], test_names[i], categories[j], tests1[i][j])
