import numpy as np
import pandas as pd

def hashFun(a,l):
	return 1

layers = 3
hashes = 10
dict[][]

def main():
	movie_df = pd.read_pickle('input_movies_vector.pkl')
	for movie_id,i in movie_df:
		
		for l in range(layers):	
			hashValue = hashFun(i,l)
			if hashValue in dict[l]:
				dict[l][hashValue].append(i)
			else:
				dict[l][hashValue] = i


