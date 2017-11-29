import numpy as np
import pandas as pd

r = 10
movie_df = pd.read_pickle('movie_matrix_svd.pkl')
movie_df = movie_df.div(np.sqrt(np.sum(np.square(movie_df), axis=1)), axis = 0)

user_input = pd.read_csv('movie_labels.csv')
train_movie = movie_df.ix[ list(user_input['movieid'])]
cosine_similarity = pd.DataFrame(np.dot(movie_df,train_movie.T),columns = train_movie.index, index = movie_df.index )


for i,row in cosine_similarity.iterrows():
	if i not in user_input:
		neighbours = row.nlargest(r).index
		neighbour_labels =  user_input[user_input['movieid'].isin(neighbours)]
		print (i,neighbour_labels['label'].value_counts().idxmax())

