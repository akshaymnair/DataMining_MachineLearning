import numpy as np
import pandas as pd
import math
from numpy.linalg import norm
import sys

layers = int(sys.argv[1])
print "Layers"
print layers
hashes = int(sys.argv[2])
print "Hashes"
print hashes
P = np.random.normal(size=[layers, hashes, 500])
w = 20
c = 0.05
b = np.random.random_sample([layers,hashes]) * w
hashtable = []
try:
	movies = sys.argv[3].split(',')
except:
	movies = []
movies = list(map(int,movies))
print "Movies"
print movies


def hashFun(a,l):
	temp = ""
	for i in xrange(hashes):
		temp += "&" + str(int(math.floor((np.dot(a,P[l][i]) + b[l][i] )/w)))
	return temp

def hashFunSimilar(a,l,dist):
	tempList = []
	for j in xrange(dist):
		temp = ""
		temp1 = ""
		for i in xrange(hashes):
			if j == i:
				temp+= "&" + str(int(math.floor((np.dot(a,P[l][i]) + b[l][i] )/w) + dist))
			else:
				temp+= "&" + str(int(math.floor((np.dot(a,P[l][i]) + b[l][i] )/w)))
		
		for i in xrange(hashes):
			if j == i:
				temp1+= "&" + str(int(math.floor((np.dot(a,P[l][i]) + b[l][i] )/w) - dist))
			else:
				temp1+= "&" + str(int(math.floor((np.dot(a,P[l][i]) + b[l][i] )/w)))
		
		if temp in hashtable[l]:
			tempList.append(temp)
		if temp1 in hashtable[l]:
			tempList.append(temp1)
	return tempList

def main():
	movie_df = pd.read_pickle('movie_matrix_svd.pkl')
	if len(movies) > 0:
		movie_df = movie_df.loc[movies]
	for l in xrange(layers):
		dict_hash = dict()
		for mid in movie_df.index:
			hash_bin = hashFun(movie_df.loc[mid].tolist(),l)
			if(hash_bin in dict_hash):
				dict_hash[hash_bin].append(mid)
			else:
				dict_hash[hash_bin] = [mid]
		hashtable.append(dict_hash)
	query_movie = int(raw_input("Enter movie id to query:- "))
	query_r = int(raw_input("Enter range of query:- "))
	movie_list = []
	for l in xrange(layers):
		hash_bin = hashFun(movie_df.loc[query_movie].tolist(),l)
		if( hash_bin in hashtable[l]):
			movie_list=  movie_list + hashtable[l][hash_bin]
	counter = 0
	while len(np.unique(movie_list)) <= query_r and counter < 4:
		if counter == 0:
			print "Probing Activated"
		for l in xrange(layers):
			for hash_bin in hashFunSimilar(movie_df.loc[query_movie],l,counter+1):
				if( hash_bin in hashtable[l]):
					movie_list =  movie_list + hashtable[l][hash_bin]
		counter = counter + 1	
		
	print "Number of unique movies considered :- " + str(len(np.unique(movie_list)))
	print "Total no of movies considered :- " + str(len(movie_list))
	
	movie_list = np.unique(movie_list)
	movie_list = np.delete(movie_list, np.argwhere(movie_list == query_movie))
	cosine_movie = pd.DataFrame(0, columns=np.unique(movie_list), index=('CosineValues',))

	for i in movie_list:
		cosine_movie[i] = np.dot(movie_df.loc[query_movie],movie_df.loc[i]) / (norm(movie_df.loc[query_movie])*norm(movie_df.loc[i]))	

	print cosine_movie.loc['CosineValues'].nlargest(query_r)
	query_vec = movie_df.loc[query_movie]

	while 1>0:
		top_movie_list = list(cosine_movie.loc['CosineValues'].nlargest(query_r).index)
		feedback = list(map(int,(raw_input("Give Feedback as String of 1(Good),0(Nuetral),-1(Bad) :- ").split(','))))
		for key,i in enumerate(feedback):
			query_vec = query_vec + i *(1.0/(feedback.count(i)+1))* movie_df.loc[movie_list[key]]
		for i in movie_list:
			cosine_movie[i] = np.dot(query_vec,movie_df.loc[i]) / (norm(query_vec)*norm(movie_df.loc[i]))
		
		print cosine_movie.loc['CosineValues'].nlargest(query_r)

		
		
	


if __name__ == '__main__':
	main()
