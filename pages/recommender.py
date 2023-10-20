import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
st.write("# Movie Recommender System")

st.write("This is a demonstration of a simple content based recommendation system. The model uses K Nearest Neighbors search to find out the movies that are most suited to the user's genre of interest.")

movies_df = pd.read_csv('pages/movies_df.csv')
ratings_df = pd.read_csv('pages/ratings_df.csv')

preferences = st.multiselect(
    'Select/Type in some movies',
    movies_df['title'])

preferences = movies_df.loc[movies_df['title'].isin(preferences)]['index'].tolist()
print('PREFERENCES')
print(preferences)
m_dict = []
num_movies = len(movies_df) 
num_users = len(ratings_df['userId'].unique())
genre_dict = {
	'Action': 0,
    'Adventure': 1,
    'Animation': 2,
    "Children": 3,
    'Comedy': 4,
    'Crime': 5,
    'Documentary': 6,
    'Drama': 7,
    'Fantasy': 8,
    'Film-Noir': 9,
    'Horror': 10,
    'Musical': 11,
    'Mystery': 12,
    'Romance': 13,
    'Sci-Fi': 14,
    'Thriller': 15,
    'War': 16,
    'Western': 17}

def unify_vec(m_vector):
  if np.sqrt(torch.matmul(m_vector, m_vector)) != 0:
    return m_vector/np.sqrt(torch.matmul(m_vector, m_vector))
  return m_vector

def make_vec_preference(genres, m_vector):
	for genre in genres:
		if genre in genre_dict.keys():
			m_vector[genre_dict[genre]] = 1
	return m_vector


def generate_recommendations():
	for i in range(num_movies):
		m_vector = [0 for i in range(len(genre_dict))]
		genres = movies_df.iloc[i]['genres'].split('|')
		m_vecor = make_vec_preference(genres, m_vector)
		m_vector = torch.tensor(m_vector)
		m_vector = unify_vec(m_vector)
		m_dict.append([movies_df.iloc[i]['movieId'], m_vector])
	vector_list = []
	for data in m_dict:
		vector_list.append(data[1].tolist())
	k = 10
	neighbors = NearestNeighbors(n_neighbors=k, algorithm='auto')
	neighbors.fit(vector_list)
	query_vector = [0 for i in range(len(genre_dict))]
	for preference in preferences:
		rec = movies_df.loc[movies_df['index'] == preference]
		for genre in rec.iloc[0][3].split('|'):
			if genre in genre_dict.keys():
				query_vector[genre_dict[genre]] = query_vector[genre_dict[genre]] + 1
	query_vector = unify_vec(torch.tensor(query_vector)).tolist()
	distances, indices = neighbors.kneighbors([query_vector])
	# print(indices)
	# print("Distances to the k-nearest neighbors:", distances[0])
	st.session_state['final_df'] = movies_df.loc[ movies_df.index.isin(indices[0]) & (~movies_df.index.isin(preferences))].sort_values(by='avg_rating', ascending=False)
st.button('Start', on_click=generate_recommendations)


if 'final_df' in st.session_state:
	st.write(st.session_state['final_df'][['title', 'avg_rating']])