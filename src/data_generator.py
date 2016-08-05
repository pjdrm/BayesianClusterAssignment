'''
Created on 03/08/2016

@author: Mota
'''
import numpy as np
from random import shuffle
import pandas as pd
import matplotlib.pyplot as plt

def gen_normal_dist_data(params, n_samples):
    label = 0
    data = []
    for mean, var in params:
        data_label_i = np.random.normal(mean, var, n_samples)
        for y in data_label_i:
            data.append([y, label])
        label += 1
    shuffle(data)
    return data
    
def sava_data(dataPath, labelsPath, data):
    dataStr = ""
    labelsStr = ""
    i = 0
    for y, label in data:
        dataStr += str(i) + "," + str(y) + "\n"
        labelsStr += str(i) + "," + str(label) + "\n"
        i += 1
    dataStr = dataStr[:-1]
    labelsStr = labelsStr[:-1]
    with open(dataPath, "w+") as f:
        f.write(dataStr)
    with open(labelsPath, "w+") as f:
        f.write(labelsStr)
        
def plot_data(dataPath):
    data = pd.Series.from_csv("data.csv")
    plt.hist(data, bins=50, histtype='stepfilled')
    plt.show()   
        
def run_data_gen(dataPath, labelsPath, params, n_samples):
    data = gen_normal_dist_data(params, n_samples)
    sava_data(dataPath, labelsPath, data)
        
        