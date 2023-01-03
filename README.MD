# Earthquake Prediction: Dictionary-Based Classifiers Code

This is a class project for USU CS 5080 (Time Series Data Mining)

It used univariate dictionary based classifiers to predict whether an earthquake would happen in a given week based on the events from previous weeks.

It contains both the source code used to run this project and a report based on the findings.

This project contains code organized into three areas:
- Merging original CSV files
- Running experiments
- Scratch page, mainly used for generating plots

## CSV Merge

This file (csv_merge.py) is to merge the original data into one CSV file. (This does not need to be done because I included the output from this file).

## Main

This is the driver for the actual experiments. (main.py)
1. It will generate time series from the experiement parameters (gen_series.py)
   - These will be saved as .npy and .pkl files for the series and targets respectively.
2. It will split the data on x data splits with stratified shuffle split and extract the features from them (split_data.py)
3. It will run experiments with each hyperparameter from a list of pre-selected hyperparameters (make_preds.py)
4. Results of each experiment is recorded into a csv file final_stats.csv

## Scratch

This was entirely for generating plots and analyzing data. I ran this for some things, but also did a lot in a command line so not every plot from my report/presentation will necessarily be represented in this code.
