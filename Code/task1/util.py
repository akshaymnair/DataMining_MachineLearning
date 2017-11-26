from datetime import datetime
import pandas as pd
from numpy import log
import os

db_folder_path = os.path.join(os.path.dirname(__file__), "..", "..", "phase3_dataset")
output_folder = os.path.join(os.path.dirname(__file__), "..", "..", "Output")

mltags_file = 'mltags.csv'
mlmovies_file = 'mlmovies.csv'
genome_tags_file = 'genome-tags.csv'
hd5_file = 'movie_final.h5'


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

def get_relevance_feedback(output_movies):
	feedback = {}
	print('Provide feedback on recommendation?')
	print('Output movies: ')
	print('%40s\t%15s\t' %('Movie id', 'Movie name'))
	for movie in output_movies:
		print ('%40s\t%15s\t' %(movie[0], movie[1]))
		feedback[movie[0]] = is_feedback_relevant()
	return feedback

def write_output_file(input_movies, output_movies, revised_movies, filename):
	f = open(os.path.abspath(os.path.join(output_folder, filename)),'w')
	f.write('For input movies: ' + '\n')
	f.write('%40s\t%15s\t\n' %('Movie id', 'Movie name'))
	for movie in input_movies:
		f.write('%40s\t%15s\t\n' %(movie[0], movie[1]))
	f.write('Output movies: \n')
	f.write('%40s\t%15s\t\n' %('Movie id', 'Movie name'))
	for movie in output_movies:
		f.write('%40s\t%15s\t\n' %(movie[0], movie[1]))
	f.write('Revised movies: \n')
	f.write('%40s\t%15s\t\n' %('Movie id', 'Movie name'))
	for movie in output_movies:
		f.write('%40s\t%15s\t\n' %(movie[0], movie[1]))

def process_output(input_movie_ids, output_movie_ids, filename):
	mlmovies = read_mlmovies()
	input_movies = get_movie_name(mlmovies, input_movie_ids)
	output_movies = get_movie_name(mlmovies, output_movie_ids)
	print_output(input_movies, output_movies)
	feedback = get_relevance_feedback(output_movies)
	
	# Do something with feedback. get revised_movies. Comment below line
	
	revised_movies = output_movies
	
	#print revised movies
	
	write_output_file(input_movies, output_movies, revised_movies, filename)

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
