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
from datetime import datetime, timedelta
import math

# input hyperparameters
n_neighbors = [3]
severity_param = 1.6



hyp_list = [n_neighbors]
combinations = list(itertools.product(*hyp_list))

dir_list = os.listdir('data')

# distance calculation function
def cal_distance (target, values, k):
    distances = []
    for val in values:
        distance = abs(target - val)
        distances.append(distance)
    nearest = heapq.nsmallest(k, distances)
    return nearest

# find the distance to the nearest neighbor
def nearest_distances(points, k=1):
    points_array = np.array(points).reshape(-1, 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(points_array)
    distances, indices = nbrs.kneighbors(points_array)
    kth_distances = distances[:, k]
    return kth_distances

# a funtion to find the turning point on any K-distance graph
def find_turning_point(sorted_list):

    minCosθ = 1
    t = None
    
    for index, value in enumerate(sorted_list):
        if index ==0 or index == len(sorted_list)-1:
            continue

        Vfp = [(index - 0)/len(sorted_list), (value - sorted_list[0])/sorted_list[-1]]
        Vpl = [(len(sorted_list) - index)/len(sorted_list), (sorted_list[-1] - value)/sorted_list[-1]]
        dot_product = Vfp[0] * Vpl[0] + Vfp[1] * Vpl[1]
        magnitude_Vfp = math.sqrt(Vfp[0] ** 2 + Vfp[1] ** 2)
        magnitude_Vpl = math.sqrt(Vpl[0] ** 2 + Vpl[1] ** 2)
        
        if magnitude_Vfp == 0 or magnitude_Vpl == 0:
            cos_theta = 1  
        else:
            cos_theta = dot_product / (magnitude_Vfp * magnitude_Vpl)

        if cos_theta < minCosθ:
            t = index
            minCosθ = cos_theta
            
    return t


# constructing a dataframe to store the achieved results.
columns = ["Loc", "F1_1", "F1_2"]
final_df = pd.DataFrame(columns=columns)

# iterating over the whole data to conduct anoaly detection (dir_list is the list of data file names)
for location in range(len(dir_list)):
    #print(location)
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
    
    all_distance = []
    weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    start_time = datetime.strptime("00:00:00", "%H:%M:%S")
    end_time = datetime.strptime("23:59:59", "%H:%M:%S")
    
    current_time = start_time
    daytime = []

    while current_time <= end_time:
        daytime.append(current_time.strftime("%H:%M:%S"))
        current_time += timedelta(minutes=15)

    for w in weekday:
        for d in daytime:
            train_loc = df_X_train[(df_X_train['Weekday'].isin([w])) & 
                                   (df_X_train['Time'].isin([d]))].copy()
            train_index = train_loc.index.tolist()
            local = X_train_main[train_index].flatten().tolist()
            train_dists = nearest_distances(local, n_neighbors[0])
            all_distance.extend(train_dists)
    
    all_distance = sorted(all_distance)
    all_distance_revised = [x for x in all_distance if x <= 3.0] # 3 refers to alpha (remove "if x <= 3.0" and the next line for SDAD-No-Cap)
    all_distance_revised.append(3.0) # the append module (de-activate this part for SDAD-No-Append)

    turning = find_turning_point(all_distance_revised)

    
    comp_point = round(all_distance_revised[turning],3)
    comp_point = comp_point * severity_param
    #print(comp_point)
    
    
    
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

            if score >= comp_point:
                predicted_labels_test.append(1)
            else:
                predicted_labels_test.append(0)

        
        answer = predicted_labels_test
        
        f1_1 = f1_score(gt_labels_1, predicted_labels_test)
        f1_2 = f1_score(gt_labels_2, predicted_labels_test) 

        new_row = [dir_list[location], f1_1, f1_2]
        final_df = final_df.append(pd.Series(new_row, index=columns), ignore_index=True)

final_df.to_csv('results/result.csv')


