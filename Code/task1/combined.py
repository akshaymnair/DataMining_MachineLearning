import pandas as pd
import sys
import util
import os
import cPickle
import numpy as np
from tensorly.decomposition import parafac
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.spatial.distance import cosine

output_file = 'task1e_combined.out.txt'
order_factor = 0.05

def tensor_decomposition():
	latent_features = 100
	actor_movie_year_3d_matrix = cPickle.load( open( "actor_movie_genre_tensor.pkl", "rb" ) )
	actor_movie_year_array = np.array(actor_movie_year_3d_matrix)
	# perform cp decomposition
	decomposed = parafac(actor_movie_year_array, latent_features, init='random')

	mlmovies = util.read_mlmovies()
	movies_list = mlmovies.movieid.unique()

	# data frame for movie factor matrix from cp decomposition
	decomposed_movies_df = pd.DataFrame(decomposed[1], index=movies_list)
	return decomposed_movies_df

def perform_svd(movie_matrix):
	no_of_components = 50
	svd = SVD(n_components=no_of_components)
	svd.fit(movie_matrix)
	svd_df = pd.DataFrame(svd.transform(movie_matrix), index=movie_matrix.index)
	return svd_df

def tensor_decomposition():
	latent_features = 10
	actor_movie_year_3d_matrix = cPickle.load( open( "actor_movie_genre_tensor.pkl", "rb" ) )
	actor_movie_year_array = np.array(actor_movie_year_3d_matrix)
	# perform cp decomposition
	decomposed = parafac(actor_movie_year_array, latent_features, init='random')
	return decomposed[1]

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	#process movie list to get matrix
	#matrix = util.get_matrix(input_movies)

	decomposed_movie_matrix = tensor_decomposition()
	
	svd_df = perform_svd(decomposed_movie_matrix)

	input_movie_df = svd_df.loc[input_movie_ids]

	#implement page rank(# of columns is 50 after performing svd)

	output_movies = []
	for index, movie in svd_df.iterrows():
		cosine_sum = 0
		order = 1
		for j, input_movie in input_movie_df.iterrows():
			cosine_sum += (1 - cosine(movie, input_movie))*order
			order -= order_factor
		output_movies.append((index, cosine_sum))
	other_movies = list(filter(lambda tup: tup[0] not in input_movie_ids, output_movies))
	other_movies.sort(key=lambda tup: tup[1], reverse=True)
	output_movie_ids = [t[0] for t in other_movies][:5]

	#print output and log them
	util.process_output(input_movie_ids, output_movie_ids, output_file)

if __name__ == "__main__":
    main()