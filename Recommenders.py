import numpy as np
import pandas as pd
import math as mt
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix #used for sparse matrix

#Class for Item-similarity Collaborative Filtering based Recommender System model

class item_similarity_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.cooccurence_matrix = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.item_similarity_recommendations = None
        
    #Get unique items (songs) corresponding to a given user
    def get_user_items(self, user):
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())
        
        return user_items
        
    #Get unique users for a given item (song)
    def get_item_users(self, item):
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())
            
        return item_users
        
    #Get unique items (songs) in the training data
    def get_all_items_train_data(self):
        all_items = list(self.train_data[self.item_id].unique())
            
        return all_items
        
    #Construct cooccurence matrix
    def construct_cooccurence_matrix(self, user_songs, all_songs):
            
        #Get users for all songs in user_songs.
        user_songs_users = []        
        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))
            
        #Initialize the item cooccurence matrix of size len(user_songs) X len(songs)
        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs), len(all_songs))), float)
           
        #Calculate similarity between user songs and all unique songs in the training data
        for i in range(0,len(all_songs)):
            #Calculate unique listeners (users) of song (item) i
            songs_i_data = self.train_data[self.train_data[self.item_id] == all_songs[i]]
            users_i = set(songs_i_data[self.user_id].unique())
            
            for j in range(0,len(user_songs)):       
                    
                #Get unique listeners (users) of song (item) j
                users_j = user_songs_users[j]
                    
                #Calculate intersection of listeners of songs i and j
                users_intersection = users_i.intersection(users_j)
                
                #Calculate cooccurence_matrix[i,j]
                if len(users_intersection) != 0:
                    #Calculate union of listeners of songs i and j
                    users_union = users_i.union(users_j)
                    
                    cooccurence_matrix[j,i] = float(len(users_intersection))/float(len(users_union))
                else:
                    cooccurence_matrix[j,i] = 0
                    
        
        return cooccurence_matrix
    
    

    
    #Use the cooccurence matrix to make top recommendations
    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs, playlistLength = 10):
        
        #Calculate a weighted average of the scores in cooccurence matrix for all user songs.
        user_sim_scores = cooccurence_matrix.sum(axis=0)/float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()
 
        #Sort the indices of user_sim_scores based upon their value and also maintain the corresponding score
        sort_index = sorted(((e,i) for i,e in enumerate(list(user_sim_scores))), reverse=True)
    
        #Create a dataframe from the following
        columns = ['user_id', 'song', 'score', 'rank']
        df = pd.DataFrame(columns=columns)
         
        #Fill the dataframe with top 10 item based recommendations
        rank = 1 
        for i in range(0,len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs and rank <= int(playlistLength):
                df.loc[len(df)]=[user,all_songs[sort_index[i][1]],sort_index[i][0],rank]
                rank = rank+1
        
        #Handle the case where there are no recommendations
        if df.shape[0] == 0:
            print("The current user has no songs for training the item similarity based recommendation model.")
            return -1
        else:
            return df
 
    #Create the item similarity based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    #Use the item similarity based recommender system model to
    #make recommendations
    def recommend(self, user):
        
        #A. Get all unique songs for this user
        user_songs = self.get_user_items(user)    
            
        print("No. of unique songs for the user: %d" % len(user_songs))
        
        #B. Get all unique items (songs) in the training data
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #D. Use the cooccurence matrix to make recommendations
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs)
                
        return df_recommendations
    
    #Get similar items to given items
    def get_similar_items(self, item_list, playlistLength):
        
        user_songs = item_list
        
        #B. Get all unique items (songs) in the training data
        all_songs = self.get_all_items_train_data()
        
        print("no. of unique songs in the training set: %d" % len(all_songs))
         
        #C. Construct item cooccurence matrix of size 
        #len(user_songs) X len(songs)
        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)
        
        #D. Use the cooccurence matrix to make recommendations
        user = ""
        df_recommendations = self.generate_top_recommendations(user, cooccurence_matrix, all_songs, user_songs, playlistLength)
         
        return df_recommendations

#Class for Matrix Factorization based Recommender System model
    
class matrix_factorization_recommender_py():
    def __init__(self):
        self.train_data = None
        self.user_id = None
        self.item_id = None
        self.songs_dict = None
        self.rev_songs_dict = None
        self.matrix_factorization_recommendations = None
        
    #Create the matrix_factorization based recommender system model
    def create(self, train_data, user_id, item_id):
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id
    
    #Compute urm
    def computeUrm(self, user_songs, user_listen_counts):
        new_user_data = {
            'song': user_songs,
            'user_id': ['nnnnn'] * len(user_songs),
            'listen_count': user_listen_counts
        }
        new_user_df = pd.DataFrame(new_user_data)
        
        # Concatenate the two DataFrames vertically to add new user data to the big DataFrame
        combined_df = pd.concat([self.train_data, new_user_df], ignore_index=True)

        # Pivot the DataFrame to create the desired matrix
        pivot_matrix = combined_df.pivot_table(index=self.user_id, columns=self.item_id, aggfunc='sum', values='listen_count', fill_value=0)

        # Convert the DataFrame to a NumPy array
        urm= pivot_matrix.values
        urm = csc_matrix(urm, dtype=np.float32)
        return urm
    
    def computeSongs_list(self):
        # Pivot the DataFrame to create the desired matrix
        pivot_matrix = self.train_data.pivot_table(index=self.user_id, columns=self.item_id, aggfunc='sum', values='listen_count', fill_value=0)

        # Convert the DataFrame to a NumPy array
        songs_list = pivot_matrix.columns.tolist()
        return songs_list
    
    #Compute SVD of the user ratings matrix
    def computeSVD(self, user_songs, user_listen_counts ):
        urm = self.computeUrm(user_songs, user_listen_counts)
        U, s, Vt = svds(urm, 2)

        dim = (len(s), len(s))
        S = np.zeros(dim, dtype=np.float32)
        for i in range(0, len(s)):
            S[i,i] = mt.sqrt(s[i])

        U = csc_matrix(U, dtype=np.float32)
        S = csc_matrix(S, dtype=np.float32)
        Vt = csc_matrix(Vt, dtype=np.float32)
        
        return U, S, Vt

    #Use the matrix_factorization based recommender system model to make recommendations
    def computeEstimatedRatings(self, user_songs, user_listen_counts, playlistLength):
        #Compute SVD of the input user ratings matrix
        U, S, Vt = self.computeSVD(user_songs, user_listen_counts)
        
        songs_list = self.computeSongs_list()
        
        rightTerm = S*Vt 
        
        #get number of users and number of songs
        max_uid = len(self.train_data['user_id'].unique())+1
        max_pid = len(self.train_data['song'].unique())
        
        userIndex = -1

        estimatedRatings = np.zeros(shape=(max_uid, max_pid), dtype=np.float16)
        
        prod = U[max_uid-1, :]*rightTerm
        #we convert the vector to dense format in order to get the indices 
        #of the songs with the best estimated ratings 
        print(estimatedRatings.shape)
        print(prod.shape)

              
        estimatedRatings[userIndex, :] = prod.todense()

        # Create a list to store the song-score pairs
        song_score_pairs = []

        # Iterate through all songs
        for i, song in enumerate(songs_list):
            # Exclude songs that are already in user_songs
            if song not in user_songs:
                song_score_pairs.append((song, estimatedRatings[max_uid-1, i]))
        
        # Sort the song-score pairs by score in descending order
        sorted_song_score_pairs = sorted(song_score_pairs, key=lambda x: x[1], reverse=True)
        
        # Create a DataFrame from the sorted song-score pairs
        df = pd.DataFrame(sorted_song_score_pairs, columns=['song', 'score'])
        df['score'] = df['score'] * 10000
        
        # Add a rank column
        df['rank'] = df.index + 1
        
        return df[:int(playlistLength)]
        


    


