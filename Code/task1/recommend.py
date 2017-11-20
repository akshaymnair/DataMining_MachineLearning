import pandas as pd
import pickle
import util

def save_obj(obj, filename):
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


df_tf_idf = util.get_movie_tf_idf_matrix()
df_movies = pd.read_csv('../../Phase2_data/mlmovies.csv')
df_tags = pd.read_csv('../../Phase2_data/genome-tags.csv')
df_users = pd.read_csv('../../Phase2_data/mlusers.csv')
df_actors = pd.read_csv('../../Phase2_data/imdb-actor-info.csv')

# List of movies, tags, users, actors
movies = df_movies.movieid.unique().tolist()
tags = df_tags.tagId.unique().tolist()
users = df_users.userid.unique().tolist()
actors = df_actors.id.unique().tolist()


# Finf tfidf of movies using tags
tf_idf = dict()
for idx, row in df_tf_idf.iterrows():
	
	max_tfidf = row.max()
	for i,r in row.iteritems():
		tf_idf[idx,i] = r / max_tfidf

movie_genre = pd.DataFrame(df_movies.genres.str.split('|').tolist(), index = df_movies.movieid).stack()
movie_genre = movie_genre.reset_index()[[0,'movieid']]
movie_genre.columns = ['genres','movieid']
genres = movie_genre.genres.unique().tolist()

user_dict = dict()
user_rating = dict()
actor_dict = dict()
actor_ranking = dict()
max_actor_rank = dict() 
df_mlratings = pd.read_csv('../../Phase2_data/mlratings.csv')
df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
for row in df_mlratings.iterrows():
	
	if row[1]['movieid'] in user_dict:
		user_dict[row[1]['movieid']].append(row[1]['userid'])

	else:
		user_dict[row[1]['movieid']] = [row[1]['userid']]
	user_rating[row[1]['movieid'], row[1]['userid']] = row[1]['rating']
df_mactors = pd.read_csv('../../Phase2_data/movie-actor.csv')

for row in df_mactors.iterrows():
	
	if row[1]['movieid'] in actor_dict:
		actor_dict[row[1]['movieid']].append(row[1]['actorid'])
		max_actor_rank[row[1]['movieid']] = max(max_actor_rank[row[1]['movieid']], (row[1]['actor_movie_rank']))
	else:
		actor_dict[row[1]['movieid']] = [row[1]['actorid']]
		max_actor_rank[row[1]['movieid']] = row[1]['actor_movie_rank']
	actor_ranking[row[1]['movieid'], row[1]['actorid']] = row[1]['actor_movie_rank']

# Given a movie id return the normalized tfidf values for all tags
def check_tag(mid):
	df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
	values = []
	for tag in tags:
		if((df_mltags[['movieid','tagid']].values == [mid, tag]).all(axis=1).any()):

			values.append(tf_idf[mid,tag])
		else:
			values.append(0)
	return values
# Given a movie id return the normalized user rating values for all users
def check_user(mid):
	df_mlratings = pd.read_csv('../../Phase2_data/mlratings.csv')
	df_mltags = pd.read_csv('../../Phase2_data/mltags.csv')
	values = []
	for user in users:
		
		if(user in user_dict[mid]):
			values.append(user_rating[mid, user] / 5.0)
		else:
			values.append(0)
	return values

# Given a movie id return the prescence or abscence values for all genres
def check_genre(mid):
	values = []
	for genre in genres:
		if((movie_genre[['movieid','genres']].values == [mid, genre]).all(axis=1).any()):
			values.append(1)
		else:
			values.append(0)
	return values

# Given a movie id return the normalized actor rank values for all actors
def check_actor(mid):
	
	values = []
	for actor in actors:
		if(actor in actor_dict[mid]):
			values.append(actor_ranking[mid, actor]/float(max_actor_rank[mid]))
		else:
			values.append(0)
	return values




cols = tags+users+genres+actors

df_movie_final = pd.DataFrame(columns=cols)

for movie_id in movies:
	tag_values = check_tag(movie_id)
	genre_values = check_genre(movie_id)
	user_values = check_user(movie_id)
	actor_values = check_actor(movie_id)
	df_movie_final.loc[movie_id] = tag_values + user_values + genre_values + actor_values
print (df_movie_final)
save_obj(df_movie_final, 'movie_final.pkl')