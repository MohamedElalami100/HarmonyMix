import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import Recommenders as Recommenders
#from Evaluation import calculate_precision

#Read userid-songid-listen_count triplets
song_df_1 = pd.read_csv('triplets_file.csv')

#Read song  metadata
song_df_2 =  pd.read_csv('song_data.csv')

#Merge the two dataframes above to create input dataframe for recommender systems
song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left") 

#Create a subset of the dataset
song_df = song_df.head(10000)

#Merge song title and artist_name columns to make a merged column
song_df['song'] = song_df['title'].map(str) + " , " + song_df['artist_name'] + song_df.apply(lambda row: f" , {row['year']}" if row['year'] != 0 else "", axis=1)
song_df = song_df.drop(columns=['song_id', 'title', 'release', 'artist_name', 'year'])

#Get all songs
allSongs = song_df['song'].unique()
allSongs_df = pd.DataFrame({'song': allSongs})

#split data into train and test data
train_data, test_data = train_test_split(song_df, test_size = 0.20, random_state=0)

#Item-Based Collaborative Filtering based Recommender System model

is_model = Recommenders.item_similarity_recommender_py()
is_model.create(train_data, 'user_id', 'song')

'''
#Call the calculate_precision function to measure precision
#This cell could take time to run
precision = calculate_precision(test_data, is_model)
print(f'Precision: {precision:.2f}')
'''

pickle.dump(is_model, open('is_model.pkl', 'wb'))

#Matrix Factorization based Recommender System model

mf_model = Recommenders.matrix_factorization_recommender_py()
mf_model.create(train_data, 'user_id', 'song')

#mf_model.computeEstimatedRatings(['U Smile - Justin Bieber','Yellow - Coldplay'], [12,5])

pickle.dump(mf_model, open('mf_model.pkl', 'wb'))




