import pandas as pd
import sys
import util
import os
import timeit
import cPickle
import numpy as np
from tensorly.decomposition import parafac
from scipy.spatial.distance import cosine

output_file = 'task1c_task2_cpd.out.txt'
no_of_components = 4
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")
order_factor = 0.05

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return
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

	output_movies = []
	# finding cosine similarity of each movie vector with the input movie vector and fetching the top 5 values
	for index, movie in decomposed_movies_df.iterrows():
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
	feedback = util.process_output(input_movie_ids, output_movie_ids, output_file)

	#process feedback to get relevant movies and movies to be excluded
	relevant_movies, movie_to_exclude = util.process_feedback(feedback, input_movie_ids)

	relevant_movie_count = len(relevant_movies)
	#if all recommended movies are relevant then return
	if relevant_movie_count==5:
		print "\nAll the movies were relevant hence no modification to the suggestion"
		return

	#fetch data frames for relevant and feedback movies
	relevant_movies_df = decomposed_movies_df.loc[relevant_movies]
	feedback_movies_df = decomposed_movies_df.loc[feedback.keys()]

	modified_query = util.probabilistic_feedback_query(feedback_movies_df, relevant_movies_df, movies_list, relevant_movie_count)

	revised_movie_ids = util.get_revised_movies(decomposed_movies_df, modified_query, movie_to_exclude)

	util.print_revised(revised_movie_ids, output_file)


if __name__ == "__main__":
    main()