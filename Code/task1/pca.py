import pandas as pd
from sklearn.decomposition import PCA
import sys
import util
import os

output_file = 'task1a_pca.out.txt'
no_of_components = 4

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	# *** Write your code here ***
	#process movie list to get matrix
	#matrix = util.get_matrix(input_movies)

	#perform PCA
	#pca = PCA(n_components=no_of_components)
	#pca.fit(matrix)

	output_movie_ids = input_movie_ids # Remove this line pass the output here

	#print output and log them
	util.write_output(input_movie_ids, output_movie_ids, output_file)

if __name__ == "__main__":
    main()
