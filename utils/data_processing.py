import pandas as pd
import numpy as np
import random

from CONSTANTS import *

## Load Data
### Ratings
ratings_path = DATA_PATH + '/ratings.dat'
ratings = pd.read_csv(ratings_path, header=None, sep='::', engine='python', names=['userId', 'movieId', 'rating', 'timestamp'])
print("ratings: \n")
print(ratings.head())

### Movies
movies_path = DATA_PATH + '/movies.dat'
movies = pd.read_csv(movies_path, header=None, sep='::', engine='python', names=['movieId', 'title', 'genres'], encoding='latin-1')
print("movies: \n")
print(movies.head())

print(f"Number of ratings: {ratings.shape[0]} \n Number of movies: {movies.shape[0]}")

## Merge Ratings and Movies
merged_data = pd.merge(ratings, movies, on='movieId')
merged_data = merged_data.sort_values(by=['timestamp'])


## Split Data
### 90% Train, 10% Validation
train_data = []
test_data = []

for user_id, user_data in merged_data.groupby('userId'):
    n_ratings = len(user_data)
    split_point = int(0.9 * n_ratings)
    train_data.append(user_data[:split_point])
    test_data.append(user_data[split_point:])

# Concatenate the user-specific train and validation sets
train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

## Randomly sample 2000 examples for validation
random_range = random.sample(range(0, 10001), 2000)
test_data = test_data.iloc[random_range]

print("Train set shape:", train_data.shape)
print("Validation set shape:", test_data.shape)


## Create Data for Sequential Modeling
### Create Test Data
def create_test_data(train_data, test_data, max_movies=10):
    """
    Processes each row of test_data and creates 'past_movies' and 'past_movie_ids' columns.

    Args:
        train_data: DataFrame containing user movie ratings history.
        test_data: DataFrame to process, containing userId and movieId.
        max_movies: Maximum number of recent movies to consider for each user.

    Returns:
        DataFrame: test_data with added 'past_movies' and 'past_movie_ids' columns.
    """

    def get_past_movies(user_id, train_data, max_movies):
        """
        Retrieves past movies watched by a user.

        Args:
            user_id: The ID of the user to look up.
            train_data: DataFrame containing movie ratings history.
            max_movies: Maximum number of recent movies to include.

        Returns:
            tuple: A list of formatted past movies and a list of corresponding movie IDs.
        """
        # Filter ratings for the specific user and sort by timestamp
        user_ratings = train_data[train_data['userId'] == user_id].sort_values('timestamp')
        
        # Get up to `max_movies` most recent movies
        recent_movies = user_ratings.tail(max_movies)
        
        # Create formatted strings for past movies
        past_movies_str = [
            f"{row['title']}:::{row['genres']}:::{row['rating']}" 
            for _, row in recent_movies.iterrows()
        ]
        
        # Extract corresponding movie IDs
        past_movie_ids = recent_movies['movieId'].tolist()
        
        return past_movies_str, past_movie_ids

    # Apply the function to each user in the test_data DataFrame
    test_data[['past_movies', 'past_movie_ids']] = test_data['userId'].apply(
        lambda user_id: pd.Series(get_past_movies(user_id, train_data, max_movies))
    )
    test_data['candidate'] = test_data['title'] + ':::' + test_data['genres']
    test_data['rating'] = test_data['rating'].astype(float)
    test_data = test_data[['userId', 'past_movies', 'past_movie_ids', 'candidate', 'movieId', 'rating']]
    return test_data

test_data_it = create_test_data(train_data, test_data, max_movies=MAX_MOVIE_SEQ_LENGTH)
print(test_data_it.head())


### Create Train Data
def create_train_data(df, max_movies=11):
    """
    Creates movie sequences for each user.
    """
    df['movie'] = df['title'] + ':::' + df['genres'] + ':::' + df['rating'].astype(str)
    user_movies = []
    for user_id, user_data in df.groupby('userId'):
        movies = user_data['movie'].tolist()
        movieIds = user_data['movieId'].tolist()
        for i in range(0, len(movies), max_movies):
            user_movies.append([user_id, len(movies[i:i + max_movies]), movies[i:i + max_movies], movieIds[i:i + max_movies]])
    df = pd.DataFrame(user_movies, columns=['userId', 'movie_count', 'past_movies', 'past_movie_ids'])
    df['candidate'] = df['past_movies'].apply(lambda x: x[-1][:-4] if x else None)
    df['movieId'] = df['past_movie_ids'].apply(lambda x: x[-1] if x else None)
    df['rating'] = df['past_movies'].apply(lambda x: x[-1][-1] if x else None).astype(float)
    df['past_movies'] = df['past_movies'].apply(lambda x: x[:-1] if x else [])
    df['past_movie_ids'] = df['past_movie_ids'].apply(lambda x: x[:-1] if x else [])
    return df

train_data_it = create_train_data(train_data, max_movies=MAX_MOVIE_SEQ_LENGTH+1)
print(train_data_it.head())
print(train_data_it.past_movie_ids.iloc[0])

## Save Data
train_data_it.to_csv(DATA_PATH + '/train_data_it.csv', index=False)
test_data_it.to_csv(DATA_PATH + '/test_data_it.csv', index=False)