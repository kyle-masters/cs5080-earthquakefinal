import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import math
import pickle

def generate_files(magnitude, weeks_to_check, intervals):
    delta = timedelta(weeks=1)/intervals
    min_date = '2011-01-02'
    max_date = '2018-07-29'
    total_weeks = math.ceil((datetime.strptime(max_date, '%Y-%m-%d') - datetime.strptime(min_date, '%Y-%m-%d')).days/7)

    # These are the 6 coordinate blocks to grab events from
    lat_min = [38, 33, 38, 37, 36, 36]
    long_min = [-123, -117, -119, -119, -118, -122]

    # Read csv file after preprocessing. Need to run csv_merge if haven't.
    df = pd.read_csv('data/events.csv', low_memory=False)

    df['time'] = df['time'].astype('datetime64')
    df['id'] = df['id'].astype('string')

    event_series = list()

    for i in range(len(lat_min)):
        event_series.append(df[(df.latitude_km >= lat_min[i]) &
                               (df.latitude_km <= lat_min[i] + 1) &
                               (df.longitude_km >= long_min[i]) &
                               (df.longitude_km <= long_min[i] + 1) &
                               (df.time >= min_date) &
                               (df.time < max_date)])
        event_series[-1] = event_series[-1][['time', 'magnitude']].to_numpy()

    targets = [list() for i in range(6)]
    events_by_week = [list() for i in range(6)]
    events_by_week_interval = np.empty((6, total_weeks, intervals), dtype=object)

    weeks = 0
    start_date = datetime.strptime(min_date, '%Y-%m-%d')
    end_date = start_date
    datetime_1 = start_date

    # Creating target and series over 395 weeks with non-sparse data
    for week in range(total_weeks):

        end_date = end_date + timedelta(weeks=1)
        # using 6 different locations from which series will be generated
        for i in range(6):
            # Find the last index of the array of series that falls in this week
            for end_i in range(event_series[i].shape[0]):
                if event_series[i][end_i][0] > end_date:
                    break

            # create new array from events falling within this week, append to by_week list and remove from original array
            events_by_week[i].append(event_series[i][:end_i])
            event_series[i] = event_series[i][end_i:]

            # calculate target as: did an event that was larger than magnitude 4 happen this week in this area?
            targets[i].append(events_by_week[i][-1][events_by_week[i][-1][:, 1] >= magnitude].shape[0] > 0)
            events_in_week = events_by_week[i][-1]
            for j in range(intervals):
                end_datetime_1 = datetime_1 + j * delta
                end_ii = 0
                for end_ii in range(len(events_in_week)):
                    if events_in_week[end_ii][0] > end_datetime_1:
                        break

                events_by_week_interval[i][week][j] = events_in_week[:end_ii, 1]
                events_in_week = events_in_week[end_ii:]

        datetime_1 = end_date

    # summing all events in a window
    features_by_interval = np.empty((6, total_weeks, intervals), dtype=tuple)
    for i in range(events_by_week_interval.shape[0]):
        for j in range(events_by_week_interval.shape[1]):
            for k in range(events_by_week_interval.shape[2]):
                if events_by_week_interval[i, j, k].shape[0] > 0:
                    features_by_interval[i, j, k] = (np.average(events_by_week_interval[i, j, k]),  # average magnitude
                                                     events_by_week_interval[i, j, k].shape[0],  # total events
                                                     np.max(events_by_week_interval[i, j, k]))  # max magnitude
                else:
                    features_by_interval[i, j, k] = (0, 0, 0)

    data_x = list()
    data_y = list()

    for week in range(weeks_to_check, total_weeks):
        for i in range(6):
            data_x.append(np.concatenate(features_by_interval[i, week-weeks_to_check:week, :], dtype=tuple))
            if len(data_x[-1]) < 10:
                data_x = data_x[:-1]
            else:
                data_y.append(targets[i][week])

    data_x = np.array(data_x, dtype=object)

    np.save(f'data_{magnitude:1.1f}_{weeks_to_check}_{intervals}_x.npy', data_x)

    with open(f'data_{magnitude:1.1f}_{weeks_to_check}_{intervals}_y.pkl', 'wb') as f:
        pickle.dump(data_y, f)
