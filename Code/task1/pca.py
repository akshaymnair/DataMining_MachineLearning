import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import sys
import util
import os

output_file = 'task1a_pca.out.txt'
no_of_components = 4
order_factor = 0.05

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	# *** Write your code here ***
	#process movie list to get matrix
	matrix = util.get_movie_matrix_from_hd5()
	#print (matrix)

	#perform PCA
	pca = PCA(n_components=no_of_components)
	pca.fit(matrix)
	pca_df = pd.DataFrame(pca.transform(matrix), index=matrix.index)

	input_movie_df = pca_df.loc[input_movie_ids]

	output_movies = []
	for index, movie in pca_df.iterrows():
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
	util.write_output(input_movie_ids, output_movie_ids, output_file)

if __name__ == "__main__":
    main()
