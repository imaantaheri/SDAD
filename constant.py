import os
import numpy as np
import pandas as pd
from extra_tools import normalize_data, df_percentage_of_ones
import itertools
import warnings
warnings.filterwarnings('ignore')
import heapq
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# input hyperparameter
n_neighbors = [3]


hyp_list = [n_neighbors]

combinations = list(itertools.product(*hyp_list))
dir_list = os.listdir('data')



def cal_distance (target, values, k):
    distances = []
    for val in values:
        distance = abs(target - val)
        distances.append(distance)
    nearest = heapq.nsmallest(k, distances)
    return nearest


def nearest_distances(points, k=1):
    points_array = np.array(points).reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points_array)
    distances, indices = nbrs.kneighbors(points_array)
    kth_distances = distances[:, k]
    return kth_distances


columns = ["Loc", "F1_1", "F1_2"]

final_df = pd.DataFrame(columns=columns)

for location in range(len(dir_list)):
    
    results = {}
    address = "data/" + dir_list[location]
    data = pd.read_csv(address)
    label_set_1 = 'label_set_1'
    contamination = df_percentage_of_ones(data, label_set_1)

    df_X_train = data.iloc[0:int(0.7 * len(data))]
    df_X_train[['Date', 'Time']] = df_X_train['Date'].str.split(expand=True)
    
    df_X_test = data.iloc[int(0.7 * len(data)):]
    df_X_test = df_X_test.reset_index(drop = True)
    gt_labels_1 = df_X_test['label_set_1'].tolist()
    gt_labels_2 = df_X_test['label_set_2'].tolist()

    gt_labels_1 = [int(i) for i in gt_labels_1]
    gt_labels_2 = [int(i) for i in gt_labels_2]

    X_train_main , X_test_main = normalize_data(data, 'Volume')
    

    for hyp in range(len(combinations)):

        predicted_labels_test = []

        for ind in range(len(X_test_main)):
            day_of_week = df_X_test['Weekday'].iloc[ind]
            time_of_day = df_X_test['Date'].iloc[ind]
            time_of_day = time_of_day[-8:]
            train_loc = df_X_train[(df_X_train['Weekday'].isin([day_of_week])) & 
                                   (df_X_train['Time'].isin([time_of_day]))].copy()
            train_index = train_loc.index.tolist()
            local = X_train_main[train_index].flatten().tolist()
            
            score = cal_distance(X_test_main[ind][0], local, combinations[hyp][0])
            
            score = score[-1]
            train_dists = nearest_distances(local, combinations[hyp][0])
            comp_point = 0.65
            if comp_point == 0:
                print(train_dists)
            if score >= comp_point:
                predicted_labels_test.append(1)
            else:
                predicted_labels_test.append(0)

        
        answer = predicted_labels_test
        
        f1_1 = f1_score(gt_labels_1, predicted_labels_test)
        f1_2 = f1_score(gt_labels_2, predicted_labels_test)

        new_row = [dir_list[location], f1_1, f1_2]
        final_df = final_df.append(pd.Series(new_row, index=columns), ignore_index=True)

final_df.to_csv('constant_results/result.csv')


