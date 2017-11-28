import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
from scipy.spatial.distance import cosine
import sys
import util
import os

output_file = 'task1b_task2_lda.out.txt'
no_of_components = 4
order_factor = 0.05


def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	#process movie list to get matrix
	matrix = util.get_movie_matrix_from_hd5()

	# *** Write your code here ***
	
	#perform LDA
	lda = LDA(n_components=no_of_components)
	lda.fit(matrix)
	lda_df = pd.DataFrame(lda.transform(matrix), index=matrix.index)

	input_movie_df = lda_df.loc[input_movie_ids]

	output_movies = []
	for index, movie in lda_df.iterrows():
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
	relevant_movies_df = lda_df.loc[relevant_movies]
	feedback_movies_df = lda_df.loc[feedback.keys()]

	modified_query = util.probabilistic_feedback_query(feedback_movies_df, relevant_movies_df, lda_df.index, relevant_movie_count)

	revised_movie_ids = util.get_revised_movies(lda_df, modified_query, movie_to_exclude)

	util.print_revised(revised_movie_ids, output_file)


if __name__ == "__main__":
    main()