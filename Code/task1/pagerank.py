import pandas as pd
import sys
import util
import os

output_file = 'task1d_pagerank.out.txt'
no_of_components = 4

def main():
	err, input_movie_ids = util.parse_input(sys.argv)
	if err:
		return

	# *** Write your code here ***
	#process movie list to get matrix
	#matrix = util.get_matrix(movie_list)

	#perform PageRank
	#output = get_page_rank()

	# Remove this line pass the output here
	output_movie_ids = input_movie_ids 

	#print output and log them
	util.write_output(input_movie_ids, output_movie_ids, output_file)

if __name__ == "__main__":
    main()