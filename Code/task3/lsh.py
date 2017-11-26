import numpy as np
import pandas as pd
import math
from numpy.linalg import norm

layers = 5
hashes = 3
P = np.random.normal(size=[layers, hashes, 500])
w = 2
b = np.random.random_sample([layers,hashes]) * w
hashtable = []

def hashFun(a,l):
	temp = ""
	count = dict()
	for i in xrange(hashes):
		temp += "&" + str(int(math.floor((np.dot(a,P[l][i]) + b[l][i] )/w)))
	return temp


def main():
	movie_df = pd.read_pickle('movie_matrix_svd.pkl')
	for l in xrange(layers):
		dict_hash = dict()
		for mid in movie_df.index:
			hash_bin = hashFun(movie_df.loc[mid].tolist(),l)
			if(hash_bin in dict_hash):
				dict_hash[hash_bin].append(mid)
			else:
				dict_hash[hash_bin] = [mid]
		hashtable.append(dict_hash)

	query_movie = 3111
	query_r = 10
	movie_list = []
	for l in xrange(layers):
		hash_bin = hashFun(movie_df.loc[query_movie].tolist(),l)
		if( hash_bin in hashtable[l] and len(hashtable[l][hash_bin])>1 ):
			movie_list=  movie_list + hashtable[l][hash_bin]

	print "Number of unique movies considered :- " + str(len(np.unique(movie_list)))
	print "Total no of movies considered :- " + str(len(movie_list))
	
	cosine_movie = pd.DataFrame(0, columns=np.unique(movie_list), index=('CosineValues',))
	for i in np.unique(movie_list):
		cosine_movie[i] = np.dot(movie_df.loc[query_movie],movie_df.loc[i]) / (norm(movie_df.loc[query_movie])*norm(movie_df.loc[i]))	

	print cosine_movie.loc['CosineValues'].nlargest(query_r+1)
	

if __name__ == '__main__':
	main()
