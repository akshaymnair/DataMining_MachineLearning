import pandas as pd
import sys
import util
import os
import cPickle
import numpy as np
from tensorly.decomposition import parafac

output_file = 'task1e_combined.out.txt'
no_of_components = 4

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

	# *** Write your code here ***
	decomposed_movie_matrix = tensor_decomposition()

	# Remove this line pass the output here
	output_movie_ids = input_movie_ids 

	#print output and log them
	util.process_output(input_movie_ids, output_movie_ids, output_file)

if __name__ == "__main__":
    main()