import pandas as pd
import sys
import util
import os
import cPickle
import numpy as np
from tensorly.decomposition import parafac

output_file = 'task1e_combined.out.txt'
no_of_components = 4

def do_tensor(input_movie_ids):
	# read the pickle file that contains tensor
	actor_movie_year_3d_matrix = cPickle.load( open( "actor_movie_genre_tensor.pkl", "rb" ) )
	actor_movie_year_array = np.array(actor_movie_year_3d_matrix)
	# perform cp decomposition
	decomposed = parafac(actor_movie_year_array, no_of_components, init='random')

	mlmovies = util.read_mlmovies()
	movies_list = mlmovies.movieid.unique()

	# data frame for movie factor matrix from cp decomposition
	decomposed_movies_df = pd.DataFrame(decomposed[1], index=movies_list)
	# dataframe containing only input movies
	input_movie_df = decomposed_movies_df.loc[input_movie_ids]

	output_movies = {}
	# finding cosine similarity of each movie vector with the input movie vector and fetching the top 5 values
	for index, movie in decomposed_movies_df.iterrows():
		cosine_sum = 0
		order = 1
		for j, input_movie in input_movie_df.iterrows():
			cosine_sum += (1 - cosine(movie, input_movie))*order
			order -= order_factor
		output_movies[index] = cosine_sum
	return output_movies

def do_svd(matrix, input_movie_ids):
	svd = SVD(n_components=no_of_components)
	svd.fit(matrix)
	svd_df = pd.DataFrame(svd.transform(matrix), index=matrix.index)

	input_movie_df = svd_df.loc[input_movie_ids]

	output_movies = {}
	for index, movie in svd_df.iterrows():
		cosine_sum = 0
		order = 1
		for j, input_movie in input_movie_df.iterrows():
			cosine_sum += (1 - cosine(movie, input_movie))*order
			order -= order_factor
		output_movies[index] = cosine_sum
	return output_movies

def do_lda(matrix, input_movie_ids):
	lda = LDA(n_components=no_of_components)
	lda.fit(matrix)
	lda_df = pd.DataFrame(lda.transform(matrix), index=matrix.index)

	input_movie_df = s_df.loc[input_movie_ids]

	output_movies = {}
	for index, movie in lda_df.iterrows():
		cosine_sum = 0
		order = 1
		for j, input_movie in input_movie_df.iterrows():
			cosine_sum += (1 - cosine(movie, input_movie))*order
			order -= order_factor
		output_movies[index] = cosine_sum
	return output_movies

def do_page_rank(input_movie_ids):
	movie_matrix_svd = pd.read_pickle('movie_matrix_svd.pkl')

	#import movies
	movie_years= []
	movies = pd.read_csv('../../phase3_dataset/mlmovies.csv')
	movie_ids = movies.movieid.unique()
	
	#Intialize PageRank values to 1.0
	pr = pd.DataFrame(1.0, columns=movie_ids, index=('PageRank',))
	
	# Calculate cosine distance between movies to movies in db
	M_sparse = sparse.csr_matrix(movie_matrix_svd)
	similarities = cosine_similarity(M_sparse)
	
	# create dataframe with headers and first colmns as movie ids
	df_movie = pd.DataFrame(similarities, columns=movie_ids, index=movie_ids)
	
	#normalize the matrix values
	norm_simil = df_movie / df_movie.sum(0)
	
	# Pagerank algorithm
	for j in range(100):
		pr_new = pr.copy()
		#Update page rank
		for i in movie_ids:
			if i in seed_movies:
				pr_new[i] = (0.15/len(seed_movies)) + (0.85*norm_simil[i].dot(pr.loc['PageRank'])) + (1/(i+1))
			else:
				pr_new[i] = (0.85*norm_simil[i].dot(pr.loc['PageRank'])) + (1/(i+1))
		pr = pr_new.copy()

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	matrix = util.get_movie_matrix_from_hd5()

	svd_list = do_svd(matrix, input_movie_ids)
	lda_list = do_lda(matrix, input_movie_ids)
	tensor_list = do_tensor(input_movie_ids)
	page_rank_list = do_page_rank(input_movie_ids)
	
	# Remove this line pass the output here
	output_movie_ids = input_movie_ids 

	#print output and log them
	util.process_output(input_movie_ids, output_movie_ids, output_file)

if __name__ == "__main__":
    main()