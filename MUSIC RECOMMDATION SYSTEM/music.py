import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Load datasets (assuming these are your existing datasets)
data = pd.read_csv('data.csv')
data_by_artist = pd.read_csv('data_by_artist.csv')
data_by_genres = pd.read_csv('data_by_genres.csv')
data_by_year = pd.read_csv('data_by_year.csv')
data_w_genres = pd.read_csv('data_w_genres.csv')


# Simulated user history (replace with actual user data if available)
def simulate_user_history(user_id):
    # Simulate user listening history
    return pd.DataFrame({
        'user_id': [user_id] * 5,
        'song_id': ['song1', 'song2', 'song3', 'song4', 'song5'],
        'play_count': [5, 3, 2, 1, 4]  # Assuming play counts within a set timeframe
    })


# Select features for recommendation
features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'valence', 'tempo',
            'duration', 'key', 'mode']

# Check if all features are present in the dataset
missing_features = [feature for feature in features if feature not in data.columns]
if missing_features:
    print(f"The following features are missing in the dataset: {missing_features}")
    # Remove missing features from the list
    features = [feature for feature in features if feature in data.columns]

# Standardize the features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Create NearestNeighbors model
nbrs = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(data_scaled)


# Function to get personalized song recommendations
def get_personalized_recommendations(user_id, data, nbrs, k=5):
    user_history = simulate_user_history(user_id)
    user_songs = user_history['song_id'].tolist()

    # Find nearest neighbors based on user listening history
    recommendations = []
    for song_id in user_songs:
        song_idx = data[data['id'] == song_id].index
        if len(song_idx) > 0:
            distances, indices = nbrs.kneighbors([data_scaled[song_idx[0]]])
            for idx in indices[0]:
                if idx != song_idx[0]:
                    song_details = data.iloc[idx][['id', 'name', 'artists']].to_dict()
                    recommendations.append(song_details)

    return recommendations[:k]


try:
    user_id = 'user123'  # Replace with an actual user ID
    recommendations = get_personalized_recommendations(user_id, data, nbrs, k=5)
    print(f"Personalized recommendations for user {user_id}:")
    for rec in recommendations:
        print(f"ID: {rec['id']}, Name: {rec['name']}, Artists: {rec['artists']}")
except KeyError as e:
    print(f"Key error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

# Display some statistics about the datasets
print("\nData statistics:")
print(data.describe())
print("\nData by artist statistics:")
print(data_by_artist.describe())
print("\nData by genres statistics:")
print(data_by_genres.describe())
print("\nData by year statistics:")
print(data_by_year.describe())
print("\nData with genres statistics:")
print(data_w_genres.describe())
