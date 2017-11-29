import pandas as pd
import sys
import os
from scipy.spatial.distance import cosine
from sklearn.decomposition import TruncatedSVD as SVD

def main():
	store = pd.HDFStore('../task1/movie_final.h5')
	movie_matrix = store['df']
	print (movie_matrix)

	svd = SVD(n_components= 500)
	svd.fit(movie_matrix)
	movie_matrix_svd = pd.DataFrame(svd.transform(movie_matrix), index=movie_matrix.index)
	
	print (movie_matrix.shape)
	print (movie_matrix_svd.shape)
	
	movie_matrix_svd.to_pickle("movie_matrix_svd.pkl")
if __name__ == "__main__":
    main()


