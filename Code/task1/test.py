import pandas as pd
import pickle
import util

df_tf_idf = util.get_movie_tf_idf_matrix()
print 'completed df_tf_idf'
df_movies = pd.read_csv('../../Dataset/mlmovies.csv')
df_tags = pd.read_csv('../../Dataset/genome-tags.csv')
df_users = pd.read_csv('../../Dataset/mlusers.csv')
df_actors = pd.read_csv('../../Dataset/imdb-actor-info.csv')
df_mlratings = pd.read_csv('../../Dataset/mlratings.csv')
df_mltags = pd.read_csv('../../Dataset/mltags.csv')
df_mactors = pd.read_csv('../../Dataset/movie-actor.csv')


# List of movies, tags, users, actors
movies = df_movies.movieid.unique().tolist()
tags = df_tags.tagId.unique().tolist()
users = df_users.userid.unique().tolist()
actors = df_actors.id.unique().tolist()

movie_genre = pd.DataFrame(df_movies.genres.str.split('|').tolist(), index = df_movies.movieid).stack()
movie_genre = movie_genre.reset_index()[[0,'movieid']]
movie_genre.columns = ['genres','movieid']
genres = movie_genre.genres.unique().tolist()

user_dict = dict()
user_rating = dict()
actor_dict = dict()
genre_dict = dict()
tags_dict = dict()
actor_ranking = dict()
max_actor_rank = dict() 

# Finf tfidf of movies using tags
tf_idf = dict()
df_tf_idf.divide(df_tf_idf.max(axis=1), axis =0)
#print df_tf_idf
	
for row in movie_genre.iterrows(): 
	if row[1]["movieid"] in genre_dict:
		genre_dict[row[1]['movieid']].append(row[1]["genres"])
	else:
		genre_dict[row[1]['movieid']] = [row[1]["genres"]]


total_length = len(df_mlratings)

count = 0
user_rating = df_mlratings.pivot(index ='movieid', columns = 'userid', values = 'rating').fillna(0)

user_rating = user_rating.divide(5.0, axis = 0)
"""for row in df_mlratings.iterrows():
	
	if row[1]['movieid'] in user_dict:
		user_dict[row[1]['movieid']].append(row[1]['userid'])

	else:
		user_dict[row[1]['movieid']] = [row[1]['userid']]
	user_rating[row[1]['movieid'], row[1]['userid']] = row[1]['rating']
	print 'completed {} row of user rating'.format((count/float(total_length))*100)
	count += 1"""


for row in df_mactors.iterrows():
	
	if row[1]['movieid'] in actor_dict:
		actor_dict[row[1]['movieid']].append(row[1]['actorid'])
		max_actor_rank[row[1]['movieid']] = max(max_actor_rank[row[1]['movieid']], (row[1]['actor_movie_rank']))
	else:
		actor_dict[row[1]['movieid']] = [row[1]['actorid']]
		max_actor_rank[row[1]['movieid']] = row[1]['actor_movie_rank']
	actor_ranking[row[1]['movieid'], row[1]['actorid']] = row[1]['actor_movie_rank']


df_mactors = df_mactors.pivot(index = 'movieid', columns = 'actorid', values = 'actor_movie_rank')

def check_user(mid):
	return user_rating.loc[mid].tolist()

def check_genre(mid):
	values = []
	for genre in genres:
		if(genre in genre_dict[mid]):
			values.append(1)
		else:
			values.append(0)
	return values

def check_tag(mid):
	return df_tf_idf.loc[mid].tolist()

def check_actor(mid):
	values = [x/max_actor_rank[mid] for x in df_mactors.loc[mid].tolist()]
	return values

cols = tags+users+genres+actors


table = []
for movie_id in movies:

	tag_values = check_tag(movie_id)
	genre_values = check_genre(movie_id)
	user_values = check_user(movie_id)
	actor_values = check_actor(movie_id)
	table.append([tag_values + user_values + genre_values + actor_values])
	print movie_id

df_movie_final = pd.DataFrame(columns=cols)
print df_movie_final

