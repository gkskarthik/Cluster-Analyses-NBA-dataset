# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 17:59:33 2018

@author: Karthik Subramanian
"""

import numpy as np
import pandas as pd
from collections import Counter

def standardize_data(Data):
    
    norm =  (Data - Data.mean(axis = 0)) / (Data.std(axis = 0))
    return norm

def test_train_split(dataFrame):

    dataFramey = dataFrame['Pos']
    dataFramex = dataFrame.drop(['Pos'],axis = 1)
    X_train = np.array(dataFramex.iloc[0:375,:])
    y_train = np.array(dataFramey.iloc[0:375])
    X_test = np.array(dataFramex.iloc[375:475,:])
    y_test = np.array(dataFramey.iloc[375:475])
    
    X_test = standardize_data(X_test)
    X_train = standardize_data(X_train)
    
    return X_train,X_test,y_train,y_test

def myknn(X_train, X_test, k):
    
    neighbors=[[]]
    dist=[]

    #For every test vector we calulate the euclidean distance between
    #test and train vectors and choose the k closest neighbours
    for i in range(0,len(X_test)):
        for j in range(0,len(X_train)):
            
            ed=np.linalg.norm(X_test[i]- X_train[j])
            dist.append(ed)
            
        a=sorted(range(len(dist)),key=lambda m:dist[m])
        neighbors.insert(i,a[:k])
        dist=[]
    
    return neighbors

def predict(y_train, y_test, neighbors, k):
    
    neighbors_pred = []
    neighbors_result = [[]]
    
    #Then we calulate the index of nearest neighbours and the get the classes
    for i in range(0,len(y_test)):
        for j in range(0, k):
            
            neighbors_pred.append(y_train[neighbors[i][j]])
            
        neighbors_result.insert(i,neighbors_pred)
        neighbors_pred=[]
    
    #print(neighbors_result)
    
    temp = []
    predict = []
    
    #print(neighbors_result)
    for i in range(0, len(y_test)):
    
        temp =  Counter(neighbors_result[i]).most_common(1)
        predict.insert(i, temp[0][0])
        temp = []
    
    count = 0
    
    for i in range(0, len(y_test)):
    
        if((predict[i] - y_test[i]) == 0):
           
            count+= 1
            
    accuracy = count/len(predict)
    
    return accuracy

data = pd.read_csv("NBAstats.csv")
data = data.drop(["Player"], axis=1)
data = data.drop(["Tm"], axis=1)
##print(data.head())

data.Pos = data.Pos.map({'C':0, 'PF':1, 'SF':2, 'SG':3, 'PG':4})
##data.head()

k = int(input("\nEnter the number of neighbors : "))

X_train,X_test,y_train,y_test = test_train_split(data)

neighbors = myknn(X_train, X_test, k)
#print(neighbors)
acc = predict(y_train, y_test, neighbors, k)

print("The accuracy for the data set is given as: ", acc)