import pandas as pd
import sys
import util
import os
import cPickle


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

	movie_year_matrix = []
	# merge the actor and movie details
	actor_movie_grouped = pd.merge(movie_actor, mlmovies, on=['movieid','movieid'], how='inner')
	# merge genre with the actor and movie details
	actor_movie_genre_grouped = pd.merge(actor_movie_grouped, movie_genre, on=['movieid','movieid'], how='inner')
	actor_movie_year_tensor = []
	count=0
	# creating the tensor
	for actor in actor_list:
		movie_year_matrix = []
		for movie in movies_list:
			movie_year_list = []
			for genre in genres:
				if actor_movie_genre_grouped[(actor_movie_genre_grouped.actorid == actor) & 
				(actor_movie_genre_grouped.movieid == movie) & (genre == actor_movie_genre_grouped.genres_y)].empty:
					movie_year_list.append(0.0)
				else:
					movie_year_list.append(1.0)
			movie_year_matrix.append(movie_year_list)
		actor_movie_year_tensor.append(movie_year_matrix)
	# store the tensor in a pickle file
	cPickle.dump( actor_movie_year_tensor, open( "actor_movie_genre_tensor.pkl", "wb" ) )

if __name__ == "__main__":
    main()