# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 03:14:26 2018

@author: Karthik Kumarasubramanian
"""

import numpy as np
import pandas as pd


def standardize_data(Data):
    norm = (Data - Data.mean(axis=0)) / (Data.std(axis=0))
    return norm


def mykmeans(Data, k):
    ##Reading data into matrix
    X = np.c_[Data]

    ##Number of rows
    n = X.shape[0]

    ##Number of columns
    p = X.shape[1]

    ##Standardizing the data at mean = 0 and Standard Deviation = 1
    X = standardize_data(X)

    ##Initializing random points 
    init_points = np.random.randint(n, size=k)

    ##Initializing Centroids
    centroids = X[init_points]

    ##Maintaining history of centroids
    hist_centroids = np.zeros((k, p))

    ##Checking the cluster assigned for each row
    cluster = np.zeros(n)

    while (hist_centroids != centroids).any():

        hist_centroids = centroids.copy()
        distances = np.zeros((n, k))

        ##Calculating the distance between points
        for i, c in enumerate(centroids):
            distances[:, i] = np.linalg.norm(X - c, axis=1)

        cluster = np.argmin(distances, axis=1)

        for j in range(k):
            X_temp = X[cluster == j]
            centroids[j] = np.apply_along_axis(np.mean, axis=0, arr=X_temp)

    return centroids, cluster


data = pd.read_csv("NBAstats.csv")
data = data.drop(["Player"], axis=1)
data = data.drop(["Tm"], axis=1)
##print(data.head())

data.Pos = data.Pos.map({'C': 0, 'PF': 1, 'SF': 2, 'SG': 3, 'PG': 4})
##data.head()

k = int(input("\nEnter the number of cluster : "))
centroids, cluster = mykmeans(data, k)
print("\n For questions 1 & 2 \n")
print("Clusters labels are \n", cluster)
print("Centroids are \n", centroids)
