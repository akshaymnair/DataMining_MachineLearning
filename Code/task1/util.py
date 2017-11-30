from datetime import datetime
import pandas as pd
from numpy import log
import os
from scipy.spatial.distance import cosine
import numpy as np

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "phase3_dataset")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

mltags_file = 'mltags.csv'
mlmovies_file = 'mlmovies.csv'
genome_tags_file = 'genome-tags.csv'
hd5_file = 'movie_final.h5'
imdb_actor_info_file = 'imdb-actor-info.csv'
movie_actor_file = 'movie-actor.csv'
movie_year_for_tensor = 2004


############### HELPER FUNCTION TO READ FILES #################################

# import mltags
def read_mltags():
	mltags =  pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mltags_file)))
	current_time = datetime.now()
	for i,row in mltags.iterrows():
		mltags.set_value(i,'timestamp', (datetime.strptime(row['timestamp'],'%Y-%m-%d %H:%M:%S') - datetime.fromtimestamp(0)).total_seconds()/
			(current_time - datetime.fromtimestamp(0)).total_seconds())
	return mltags

# import mlmovies
def read_mlmovies():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, mlmovies_file)))

# import genome-tags
def read_genome_tags():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, genome_tags_file)))

# import imdb-actor-info
def read_imdb_actor_info():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, imdb_actor_info_file)))

# import movie-actor
def read_movie_actor():
	return pd.read_csv(os.path.abspath(os.path.join(db_folder_path, movie_actor_file)))

#To implement
def get_matrix(movie_list):
	return None

def get_movie_name(mlmovies, movie_ids):
	return [tuple(x) for x in mlmovies[mlmovies.movieid.isin(movie_ids)].to_records(index=False)]

def parse_input(args):
	if len(args) < 2:
		print('Expected arguments are not provided.')
		return True, None
	input_movie_ids = []
	for i in range(1, len(args)):
		input_movie_ids.append(int(args[i]))
	return False, input_movie_ids

def is_feedback_relevant():
	ufb = raw_input('Is movie relevant to you? (Type N/n for no): ')
	if ufb.lower() == 'n' or ufb.lower() == 'no':
		return False
	return True

def print_output(input_movies, output_movies):
	print('For input movies: ')
	print('%40s\t%15s\t' %('Movie id', 'Movie name'))
	for movie in input_movies:
		print ('%40s\t%15s\t' %(movie[0], movie[1]))
	print('Output movies: ')
	print('%40s\t%15s\t' %('Movie id', 'Movie name'))
	for movie in output_movies:
		print ('%40s\t%15s\t' %(movie[0], movie[1]))

def print_revised(revised_movie_ids, filename):
	mlmovies = read_mlmovies()
	revised_movies = get_movie_name(mlmovies, revised_movie_ids)
	print('Revised movies: ')
	print('%40s\t%15s\t' %('Movie id', 'Movie name'))
	for movie in revised_movies:
		print ('%40s\t%15s\t' %(movie[0], movie[1]))
	write_revised(revised_movies, filename)

def write_revised(revised_movies, filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'a')
	f.write('Revised movies: \n')
	f.write('%40s\t%15s\t\n' %('Movie id', 'Movie name'))
	for movie in revised_movies:
		f.write('%40s\t%15s\t\n' %(movie[0], movie[1]))

def get_relevance_feedback(output_movies):
	feedback = {}
	print('Provide feedback on recommendation?')
	print('Output movies: ')
	print('%40s\t%15s\t' %('Movie id', 'Movie name'))
	for movie in output_movies:
		print ('%40s\t%15s\t' %(movie[0], movie[1]))
		feedback[movie[0]] = is_feedback_relevant()
	return feedback

def write_output_file(input_movies, output_movies, filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'w')
	f.write('For input movies: ' + '\n')
	f.write('%40s\t%15s\t\n' %('Movie id', 'Movie name'))
	for movie in input_movies:
		f.write('%40s\t%15s\t\n' %(movie[0], movie[1]))
	f.write('Output movies: \n')
	f.write('%40s\t%15s\t\n' %('Movie id', 'Movie name'))
	for movie in output_movies:
		f.write('%40s\t%15s\t\n' %(movie[0], movie[1]))

def process_output(input_movie_ids, output_movie_ids, filename):
	mlmovies = read_mlmovies()
	input_movies = get_movie_name(mlmovies, input_movie_ids)
	output_movies = get_movie_name(mlmovies, output_movie_ids)
	print_output(input_movies, output_movies)
	feedback = get_relevance_feedback(output_movies)
	revised_movies = output_movies
	
	write_output_file(input_movies, output_movies, filename)
	
	return feedback

################## HELPER FUNCTION TO PROCESS AND RETRIEVE ######################

def get_movie_matrix_from_hd5():
	store = pd.HDFStore(hd5_file)
	return store['df']

def get_movie_tf_idf_matrix():
	mltags = read_mltags()

	movie_count = mltags.loc[:,'movieid'].unique().shape[0]

	# Needed for TF denominators. Calcuate total number of tags per movie
	tags_per_movie = mltags.groupby('movieid', as_index=False)['tagid'].agg({'m_count' : pd.Series.count})

	# Needed for IDF denonminators. Calculate unique movieids for each tag
	movies_per_tag = mltags.groupby('tagid', as_index=False)['movieid'].agg({'t_count' : pd.Series.nunique})

	# Grouped so as to create unique tagid, movieid pairs and calculate term frequency from timestamp
	tagid_movieid_grouped = mltags.groupby(['tagid', 'movieid'], as_index=False)['timestamp'].agg({'tf': 'sum'})

	# Merge tag_counts. Add new column including calculated tags_per_movie.
	M1 = pd.merge(tagid_movieid_grouped, tags_per_movie, on=['movieid','movieid'], how='inner')

	# Merge movie_counts. Add new column including calculated movies_per_tag.
	M2 = pd.merge(M1, movies_per_tag, on=['tagid', 'tagid'], how = 'inner')

	# Perform TF-IDF from the data.
	M2['tfidf'] = M2['tf']*log(movie_count/M2['t_count'])/M2['m_count']
	#print M2

	# Pivot the matrix to get in required form 
	R = M2.pivot(index='movieid', columns='tagid', values='tfidf').fillna(0)
	#print(R)
	return R

	# Pivot the matrix to get in required form 
	R = M.pivot(index='movieid', columns='tagid', values='tfidf').fillna(0)
	#print (R)
	return R

def process_feedback(feedback, input_movie_ids):
	movie_to_exclude = input_movie_ids
	# Do something with feedback. get revised_movies.
	relevant_movies = []
	irrelevant_movies = []
	for movie_id in feedback:
		if feedback[movie_id]==True:
			relevant_movies.append(movie_id)
		else:
			# to remove irrelevant movies from recommendation, add the irrelevant movie to the list
			movie_to_exclude.append(movie_id)
	return relevant_movies, movie_to_exclude

def probabilistic_feedback_query(feedback_movies_df, relevant_movies_df, movies_list, relevant_movie_count):
	# probabilistic feedback query calculation
	n_i = feedback_movies_df.sum()
	r_i = relevant_movies_df.sum()
	N = len(movies_list)
	R = relevant_movie_count
	log_numerator = (r_i + (n_i / N))/(R - r_i + 1)
	log_denominator = (n_i - r_i + (n_i / N))/(N - R - n_i + r_i + 1)
	modified_query = np.log(log_numerator/log_denominator)
	modified_query = modified_query.fillna(value=0)
	return modified_query

def get_revised_movies(movies_df, modified_query, input_movie_ids):
	# finding cosine similarity of modified query with the input movie vector and fetching the top 5 values
	revised_movies = []
	for index, movie in movies_df.iterrows():
		cosine_similarity = (1 - cosine(movie, modified_query))
		revised_movies.append((index, cosine_similarity))
	other_movies = list(filter(lambda tup: tup[0] not in input_movie_ids, revised_movies))
	other_movies.sort(key=lambda tup: tup[1], reverse=True)
	revised_movie_ids = [t[0] for t in other_movies][:5]
	return revised_movie_ids

def get_similarity(movies_df, modified_query):
	# finding cosine similarity of modified query with the input movie vector and fetching the top 5 values
	movie_similarity_list = {}
	for index, movie in movies_df.iterrows():
		cosine_similarity = (1 - cosine(movie, modified_query))
		movie_similarity_list[index] = cosine_similarity
	return movie_similarity_list

def get_revised_movies_combined(combined_similarity, input_movie_ids):
	other_movies = list(filter(lambda tup: tup[0] not in input_movie_ids, combined_similarity))
	other_movies.sort(key=lambda tup: tup[1], reverse=True)
	revised_movie_ids = [t[0] for t in other_movies][:5]
	return revised_movie_ids

