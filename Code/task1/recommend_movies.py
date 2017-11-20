import pickle
from sklearn.metrics import pairwise_distances as Distance
import pandas as pd
import sys
def load_obj(file):
	with open(file,'rb') as input:
		obj = pickle.load(input)
	return obj


# Load dataframe that was saved by runnning recommend.py
df = load_obj('movie_final.pkl')
# Default user id is 8
user = 8
if(len(sys.argv)>1):
	user = int(sys.argv[1])

# Find the movie ids which the given user has tagged or rated and assume it as the movies watched by the user
df_mlratings = pd.read_csv('../../Phase2_data/mlratings.csv')
df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')

movie_ids = df_mlratings.loc[df_mlratings['userid']== user].movieid.unique().tolist()
movie_ids = movie_ids + df_mltags.loc[df_mltags['userid']==user].movieid.unique().tolist()

if(movie_ids==[]):
	print ('The user has not watched any movie')
else:

	# Calculate cosine distance between movies watched by the user and all the other movies and store it in a matrix D
	X = df.loc[movie_ids]
	Y = df
	D = Distance(X,Y, metric='cosine')

	# Find the sum of distances of each movie to every movie watched by user
	D_sum = D.sum(axis = 0)


	index_order = reversed(D_sum.argsort())

	count = 0 
	recommend_movie_ids = []
	for index1 in index_order:
		if(df.index[index1] not in movie_ids):
			recommend_movie_ids.append(df.index[index1])
			count = count + 1
		if count == 5:
			break
		

	print ('--------------Movies watched by users----------------')

	df_movies = pd.read_csv('../../Phase2_data/mlmovies.csv')

	for mid in  movie_ids:
		print( df_movies.loc[df_movies['movieid']==mid,'moviename'].iloc[0])

	print( '---------------Recommended movies--------------------')

	for mid in  recommend_movie_ids:
		print(mid)
		print( df_movies.loc[df_movies['movieid']==mid,'moviename'].iloc[0])