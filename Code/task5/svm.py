# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 07:07:56 2017

@author: supra
"""

import csv
import pickle
import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

class Support_Vector_Machine:
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]
        
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)

        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None

        # support vectors yi(xi.w+b) = 1
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,
                      ]
        
        b_range_multiple = 2
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
        
        
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
            
                        # yi(xi.w+b) >= 1
                        for i in self.data:
                            for xi in self.data[i]:
                                yi=i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    #print(xi,':',yi*(np.dot(w_t,xi)+b))
                                    
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]
                
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step
                    
                    
            norms = sorted([n for n in opt_dict])
            #||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2       
            
        for i in self.data:
            for xi in self.data[i]:
                yi=i
                print(xi,':',yi*(np.dot(self.w,xi)+self.b))  
    
    
    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification   
    

def load_obj(file):
	with open(file,'rb') as input:
		obj = pickle.load(input)
	return obj


def parse_train_input():
    df = load_obj('movie_matrix_svd.pkl')
    #print(df)
    with open ('Training-data.csv') as data:
      reader = csv.reader(data)  
      for row in reader:
          movieId = int(row[0])
          label = row[1]
          #print(label)
          #print(movieId)
          for index, row in df.iterrows():
              if(index == movieId):
                  feature1 = float(row[0])
                  feature2 = float(row[1])
                  #print(feature1,feature2)
                  if label in data_dict.keys():
                      data_dict[label].append((feature1,feature2))
                  else:
                      data_dict[label] = [(feature1,feature2)]
    print(data_dict)
                
    
svm = Support_Vector_Machine()
svm.fit(data = data_dict)

predict_us = [[5.78, 2.3587],[0.4587,5.2548],[47.5489,38.25484][19.2548,23.2548]]

for p in predict_us:
    svm.predict(p)
