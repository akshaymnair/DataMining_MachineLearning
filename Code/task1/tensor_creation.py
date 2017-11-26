import pandas as pd
import sys
import util
import os
import cPickle
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial


def main():
	# load the required data from csv files
	mlmovies = util.read_mlmovies()
	imdb_actor_info = util.read_imdb_actor_info()
	movie_actor = util.read_movie_actor()
	movies_list = mlmovies.movieid.unique()
	actor_list = imdb_actor_info.id.unique()

	# split the '|' separated genre values and stack them on separate columns
	movie_genre = pd.DataFrame(mlmovies.genres.str.split('|').tolist(), index = mlmovies.movieid).stack()
	movie_genre = movie_genre.reset_index()[[0,'movieid']]
	movie_genre.columns = ['genres','movieid']
	genres = movie_genre.genres.unique().tolist()

	# merge the actor and movie details
	actor_movie_grouped = pd.merge(movie_actor, mlmovies, on=['movieid','movieid'], how='inner')
	# merge genre with the actor and movie details
	actor_movie_genre_grouped = pd.merge(actor_movie_grouped, movie_genre, on=['movieid','movieid'], how='inner')
	actor_movie_genre_grouped.sort_values(['actorid', 'movieid'], ascending=[1, 1])
	actor_movie_genre_tensor = [None] * len(actor_list)
	
	actor_dict = {}
	for i in range(0,len(actor_list)):
		actor_dict[actor_list[i]] = i
	movie_dict = {}
	for i in range(0,len(movies_list)):
		movie_dict[movies_list[i]] = i
	genre_dict = {}
	for i in range(0,len(genres)):
		genre_dict[genres[i]] = i

	actor_movie_genre_tensor = [None] * len(actor_list)
	for i in range(0,len(actor_list)):
		actor_movie_genre_tensor[i] = [None] * len(movies_list)
		for  j in range(0,len(movies_list)):
			actor_movie_genre_tensor[i][j] = [None] * len(genres)
			for k in range(0,len(genres)):
				actor_movie_genre_tensor[i][j][k] = 0.0

	for index, row in actor_movie_genre_grouped.iterrows():
		a_id=row['actorid']
		m_id=row['movieid']
		g_id=row['genres_y']
		actor_movie_genre_tensor[actor_dict[a_id]][movie_dict[m_id]][genre_dict[g_id]]  = 1.0

	cPickle.dump( actor_movie_genre_tensor, open( "actor_movie_genre_tensor.pkl", "wb" ) )

if __name__ == "__main__":
    main()