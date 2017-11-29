#author- Akshay
from __future__ import print_function
import numpy
import pickle
import pandas as pd
import sys
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import numpy as np
import util

def main():
	
	# get user input, movies watched by the user
	seed_movies = sys.argv[1].split(',')
	seed_movies = list(map(int, seed_movies))
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
	for j in range(0,5):
		pr_new = pr.copy()
		#Update page rank
		for i in movie_ids:
			if i in seed_movies:
				pr_new[i] = (0.15/len(seed_movies)) + (0.85*norm_simil[i].dot(pr.loc['PageRank'])) + (0.01/(seed_movies.index(i)+1))
			else:
				pr_new[i] = (0.85*norm_simil[i].dot(pr.loc['PageRank'])) 
		pr = pr_new.copy()

	#Remove seeded actors from list
	for i in seed_movies:
		try:
			pr_final = pr.drop(i ,axis=1) 
			pr = pr_final
		except:
			pass
	#Display top 5 related actors
	print("\nMovies user watched: ", seed_movies)
	
	O_file = 'task1d_task2_pagerank.out.txt'
	r_movies = [ ]
	r_movies = list(pr.loc['PageRank'].nlargest(5).index)

	feedback = util.process_output(seed_movies, r_movies, O_file)

	#process feedback to get relevant movies and movies to be excluded
	relevant_movies, movie_to_exclude = util.process_feedback(feedback, seed_movies)
	irrelevant_movies = np.setdiff1d(movie_to_exclude, seed_movies)

	relevant_movie_count = len(relevant_movies)
	#if all recommended movies are relevant then return
	if relevant_movie_count == 5:
		print ("All the movies were relevant hence no modification to the suggestion")
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

	#Remove seeded actors from list
	for i in seed_movies:
		try:
			pr_final = pr.drop(i ,axis=1) 
			pr = pr_final
		except:
			pass
	#Display top 5 related actors
	
	r_movies = [ ]
	r_movies = list(pr.loc['PageRank'].nlargest(5).index)

	util.print_revised(r_movies, O_file)


if __name__ == "__main__":
    main()

#END