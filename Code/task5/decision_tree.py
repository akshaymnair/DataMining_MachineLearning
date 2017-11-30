#author - Akshay
from __future__ import print_function
import pickle
import pandas as pd
import numpy
import sys
import os

#Driver funciton: read user input, database, call tree functions
def main():
	movies_features = pd.read_pickle('movie_matrix_svd.pkl')    
	user_input = pd.read_csv('movie_labels.csv')
	movies = pd.read_csv('../../phase3_dataset/mlmovies.csv')
	movie_ids = movies.movieid.unique()
	movies_features['movieid'] = pd.Series(movie_ids, index=movies_features.index) # add a new column
	u_input = pd.DataFrame(user_input)
	merged = pd.merge(movies_features,u_input, on=['movieid'])
	merged = merged.drop('movieid', axis=1)
	train_data =  merged.values
	my_tree = build_tree(train_data)
	test_data = movies_features.values
	f = open('../../Output/task5b_DecisionTree.txt','w')
	for row in test_data:
		print ("Movie ID: %s is Labled: %s" %(int(row[-1]), parse(classify(row, my_tree))))
		f.write(str("\nMovie ID: %s is Labled: %s" %(int(row[-1]), parse(classify(row, my_tree)))))

# count number of differnet labels in trining set
def class_counts(rows):
    counts = {} 
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

#condition to split dataset
class condition:    
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        if (val > self.value):
            return True
        else:
            return False
            
#put date from the test data to two splits according to the results from 'condition', true in one group, false in another group
def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows

#calculate gini_impurity :https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
def gini(rows):  
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity

#calculate informatoin gain: https://en.wikipedia.org/wiki/Decision_tree_learning
def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

#iterate over all feature and calculate info gain and decide best condition to split upon
def split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1 
    for col in range(n_features): 
        values = set([row[col] for row in rows])
        for val in values: 
            question = condition(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question

#stores dictionary of labels and number of times it is reached in each row
class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)

#saves the condition and answers
class D_Node:
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch

#recursively build tree
def build_tree(rows):
    gain, question = split(rows)
    if gain == 0:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return D_Node(question, true_branch, false_branch)

# get value from the true o false branch
def classify(row, node):   
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)

# get the label names from the node dictionary
def parse(counts):
    for lbl in counts.keys():
    	label = lbl
    return label

if __name__ == '__main__':
	main()
#END
