import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation as LDA
import sys
import util
import os

output_file = 'task1b_lda.out.txt'
no_of_components = 4

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	#process movie list to get matrix
	#matrix = util.get_matrix(input_movies)

	# *** Write your code here ***
	
	#perform LDA
	#lda = LDA(n_components=no_of_components)
	#lda.fit(matrix)

	# Remove this line pass the output here
	output_movie_ids = input_movie_ids 

	#print output and log them
	util.write_output(input_movie_ids, output_movie_ids, output_file)

if __name__ == "__main__":
    main()