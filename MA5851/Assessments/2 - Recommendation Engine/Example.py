# import required modules
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
#import sys
#import pickle

os.getcwd()
os.chdir(r"D:\Libraries\Documents\Data Science\JCU_MDS\2021_Masterclass 1\Assignment 2_copy\Reference Material\MovieLens database")

# read in the data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
tags = pd.read_csv('tags.csv')
# genome_scores = pd.read_csv('genome-scores.csv')
# genome_tags = pd.read_csv('genome-tags.csv')

movies.tail()
movies['genres'] = movies['genres'].str.replace('|',' ', regex=True)

# filter and clean the data
ratings_f = ratings.groupby('userId').filter(lambda x: len(x) >= 2000)
movie_list_rating = ratings_f.movieId.unique().tolist()
movies = movies[movies.movieId.isin(movie_list_rating)]

Mapping_file = dict(zip(movies.title.tolist(), movies.movieId.tolist()))

tags.drop(['timestamp'],1, inplace=True)
ratings_f.drop(['timestamp'],1, inplace=True)

# merge movies and tags dataframes and create a metadata tag for each movie
mixed = pd.merge(movies, tags, on='movieId', how='left')
mixed.head(3)

mixed.fillna("", inplace=True)
mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: "%s" % ' '.join(x)))
Final = pd.merge(movies, mixed, on='movieId', how='left')
Final ['metadata'] = Final[['tag', 'genres']].apply(lambda x: ' '.join(x), axis = 1)
Final[['movieId','title','metadata']].head(3)

# create a collaborative latent matrix from user ratings
ratings_f.head()
ratings_f1 = pd.merge(movies[['movieId']], ratings_f, on="movieId", how="right")

ratings_f2 = ratings_f1.pivot(index = 'movieId', columns ='userId', values = 'rating').fillna(0)
ratings_f2.head(3)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=200)
latent_matrix_2 = svd.fit_transform(ratings_f2)
latent_matrix_2_df = pd.DataFrame(latent_matrix_2, index=Final.title.tolist())

explained = svd.explained_variance_ratio_.cumsum()
plt.plot(explained, '.-', ms = 16, color='red')
plt.xlabel('Singular value components', fontsize= 12)
plt.ylabel('Cumulative percent of variance', fontsize=12)
plt.show()

# computing cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
a_2 = np.array(latent_matrix_2_df.loc['Toy Story (1995)']).reshape(1, -1)

# calculate the similartity of this movie with the others in the list
score_2 = cosine_similarity(latent_matrix_2_df, a_2).reshape(-1)

# form a data frame of similar movies
dictDf = {'collaborative': score_2}
similar = pd.DataFrame(dictDf, index = latent_matrix_2_df.index )

similar.sort_values('collaborative', ascending=False, inplace=True)

similar[1:].head(11)