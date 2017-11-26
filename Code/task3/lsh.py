import numpy as np
import pandas as pd
import math

layers = 50
hashes = 10
P = np.random.normal(size=[layers, hashes, 500])
w = 2
b = np.random.random_sample([layers,hashes]) * w


hashtable = []





def setInitalP():
	np.random.normal()
def hashFun(a,l):
	temp = ""
	count = dict()
	for i in xrange(hashes):
		temp += str(int(math.floor((np.dot(a,P[l][i]) + b[l][i] )/w)))
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

	q_mid = 3111
	print 'Printing movies in the same bin as query movie', q_mid

	for l in xrange(layers):
		hash_bin = hashFun(movie_df.loc[q_mid].tolist(),l)
		print hash_bin
		if( hash_bin in hashtable[l] and len(hashtable[l][hash_bin])>1 ):
			print hashtable[l][hash_bin]

		else:
			print 'No movies in the bin for layer', l
	'''for hdict in hashtable:
		for key in hdict:
			if(len(hdict[key])>1):
				print hdict[key]'''





if __name__ == '__main__':
	main()