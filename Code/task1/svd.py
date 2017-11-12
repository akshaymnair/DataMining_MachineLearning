import pandas as pd
from sklearn.decomposition import TruncatedSVD as SVD
import sys
import util
import os

output_file = 'task1a_svd.out.txt'
no_of_components = 4

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	# *** Write your code here ***
	#process movie list to get matrix
	#matrix = util.get_matrix(input_movies)

	#perform SVD
	#svd = SVD(n_components=no_of_components)
	#svd.fit(matrix)

	# Remove this line pass the output here
	output_movie_ids = input_movie_ids 

	#print output and log them
	util.write_output(input_movie_ids, output_movie_ids, output_file)

if __name__ == "__main__":
    main()