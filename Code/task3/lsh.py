import numpy as np
import pandas as pd

layers = 3
hashes = 10
P = np.random.normal(size=[layers, hashes, 500])
w = 1000
b = np.random.random_sample([layers,hashes]) * w



'''
def setInitalP():
	np.random.normal()
def hashFun(a,l):
	for i in range(l[k]):
		np.dot(a,P[l][k]) + b[l][k])/w[l]

def main():
	movie_df = pd.read_pickle('input_movies_vector.pkl')
	for movie_id,i in movie_df:
		
		for l in range(layers):	
			
			hashValue = hashFun(i,l)
			if hashValue in dict[l]:
				dict[l][hashValue].append(i)
			else:
				dict[l][hashValue] = i

'''
