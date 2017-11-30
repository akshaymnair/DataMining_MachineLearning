import pandas as pd
import sys
import util
import os
import cPickle
import pickle

import numpy as np
from scipy import sparse
from tensorly.decomposition import parafac
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.metrics.pairwise import cosine_similarity

output_file = 'task1e_combined.out.txt'
no_of_components = 4
order_factor = 0.05

def do_tensor(input_movie_ids):
	# read the pickle file that contains tensor
	actor_movie_year_3d_matrix = cPickle.load( open( "actor_movie_genre_tensor.pkl", "rb" ) )
	actor_movie_year_array = np.array(actor_movie_year_3d_matrix)
	# perform cp decomposition
	decomposed = parafac(actor_movie_year_array, no_of_components, init='random')

	mlmovies = util.read_mlmovies()
	mlmovies = mlmovies.loc[mlmovies['year'] >= util.movie_year_for_tensor]
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
	return output_movies, decomposed_movies_df, movies_list

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
	return output_movies, svd_df

def do_lda(matrix, input_movie_ids):
	lda = LDA(n_components=no_of_components)
	lda.fit(matrix)
	lda_df = pd.DataFrame(lda.transform(matrix), index=matrix.index)

	input_movie_df = lda_df.loc[input_movie_ids]

	output_movies = {}
	for index, movie in lda_df.iterrows():
		cosine_sum = 0
		order = 1
		for j, input_movie in input_movie_df.iterrows():
			cosine_sum += (1 - cosine(movie, input_movie))*order
			order -= order_factor
		output_movies[index] = cosine_sum
	return output_movies, lda_df

def do_page_rank(seed_movies):
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
	for j in range(5):
		pr_new = pr.copy()
		#Update page rank
		for i in movie_ids:
			if i in seed_movies:
				pr_new[i] = (0.15/len(seed_movies)) + (0.85*norm_simil[i].dot(pr.loc['PageRank'])) + (1/(i+1))
			else:
				pr_new[i] = (0.85*norm_simil[i].dot(pr.loc['PageRank'])) + (1/(i+1))
		pr = pr_new.copy()

	output_movies = pd.Series(pr.loc['PageRank'], index=pr.columns.values).to_dict()
	return output_movies

def do_page_rank_relevance(feedback, seed_movies):
	movie_matrix_svd = pd.read_pickle('movie_matrix_svd.pkl')

	movies = pd.read_csv('../../phase3_dataset/mlmovies.csv')
	movie_ids = movies.movieid.unique()

	#process feedback to get relevant movies and movies to be excluded
	relevant_movies, movie_to_exclude = util.process_feedback(feedback, seed_movies)
	irrelevant_movies = np.setdiff1d(movie_to_exclude, seed_movies)

	# Calculate cosine distance between movies to movies in db
	M_sparse = sparse.csr_matrix(movie_matrix_svd)
	similarities = cosine_similarity(M_sparse)
	
	# create dataframe with headers and first colmns as movie ids
	df_movie = pd.DataFrame(similarities, columns=movie_ids, index=movie_ids)

	#normalize the matrix values
	norm_simil = df_movie / df_movie.sum(0)

	relevant_movie_count = len(relevant_movies)
	#if all recommended movies are relevant then return
	if relevant_movie_count == 5:
		print ("\nAll the movies were relevant hence no modification to the suggestion")
		return

	pr = pd.DataFrame(1.0, columns=movie_ids, index=('PageRank',))
	# Pagerank algorithm
	for j in range(0,5):
		pr_new = pr.copy()
		#Update page rank
		for i in movie_ids:
			if i in seed_movies:
				pr_new[i] = (0.15/len(seed_movies)) + (0.85*norm_simil[i].dot(pr.loc['PageRank'])) + (0.01/(seed_movies.index(i)+1))
			else:
				pr_new[i] = (0.85*norm_simil[i].dot(pr.loc['PageRank'])) 
			if i in irrelevant_movies:
				pr_new[i] = pr_new[i] - 0.02
			elif i in relevant_movies:
				pr_new[i] = pr_new[i] + 0.02
		pr = pr_new.copy()

	output_movies = pd.Series(pr.loc['PageRank'], index=pr.columns.values).to_dict()
	return output_movies

def process_feedback_movie_vector(feedback, input_movie_ids, movies_list, movies_df):
	#process feedback to get relevant movies and movies to be excluded
	relevant_movies, movie_to_exclude = util.process_feedback(feedback, input_movie_ids)

	relevant_movie_count = len(relevant_movies)
	#if all recommended movies are relevant then return
	if relevant_movie_count==5:
		print "\nAll the movies were relevant hence no modification to the suggestion"
		return

	#fetch data frames for relevant and feedback movies
	relevant_movies_df = movies_df.loc[relevant_movies]
	feedback_movies_df = movies_df.loc[feedback.keys()]

	modified_query = util.probabilistic_feedback_query(feedback_movies_df, relevant_movies_df, movies_list, relevant_movie_count)

	similarity_matrix = util.get_similarity(movies_df, modified_query)

	return similarity_matrix, movie_to_exclude

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	matrix = util.get_movie_matrix_from_hd5()

	svd_dict, svd_df = do_svd(matrix, input_movie_ids)
	print('SVD done')
	lda_dict, lda_df = do_lda(matrix, input_movie_ids)
	print('LDA done')
	tensor_dict, decomposed_movies_df, movies_list = do_tensor(input_movie_ids)
	print('Tensor done')
	page_rank_dict = do_page_rank(input_movie_ids)
	print('PageRank done')

	other_movies = []
	for k in svd_dict:
		if k in input_movie_ids:
			continue
		total_weight = svd_dict[k] + lda_dict.get(k, 0) + tensor_dict.get(k, 0) + page_rank_dict.get(k, 0)
		other_movies.append((k, total_weight))

	other_movies.sort(key=lambda tup: tup[1], reverse=True)
	output_movie_ids = [t[0] for t in other_movies][:5]

	feedback = util.process_output(input_movie_ids, output_movie_ids, output_file)

	similarity_svd, movie_to_exclude = process_feedback_movie_vector(feedback, input_movie_ids, svd_df.index, svd_df)

	similarity_lda, movie_to_exclude = process_feedback_movie_vector(feedback, input_movie_ids, lda_df.index, lda_df)

	similarity_tensor, movie_to_exclude = process_feedback_movie_vector(feedback, input_movie_ids, movies_list, decomposed_movies_df)

	page_rank_dict = do_page_rank_relevance(feedback, input_movie_ids)
	
	other_movies = []
	for k in similarity_svd:
		if k in movie_to_exclude:
			continue
		total_weight = similarity_svd[k] + similarity_lda.get(k, 0) + similarity_tensor.get(k, 0) + page_rank_dict.get(k, 0)
		other_movies.append((k, total_weight))

	other_movies.sort(key=lambda tup: tup[1], reverse=True)
	revised_movie_ids = [t[0] for t in other_movies][:5]

	util.print_revised(revised_movie_ids, output_file)



if __name__ == "__main__":
    main()