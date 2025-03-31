import utils
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os


def check_token(user_id):
    try:
        query = '''
            SELECT * FROM users
            WHERE user_id = %s
        '''

        engine = utils.get_engine()
        user_data = pd.read_sql(query, engine, params=(user_id,))
        if len(user_data) == 0:
            return (False, user_id)
        else:
            return (True, user_id)
    except Exception as e:
        print(f'Error checking token: {e}')
        return False, None


def recommend_songs(user_profile_vector: list, song_features: pd.DataFrame, top_n=10):

    user_profile_vector = np.array(user_profile_vector).reshape(1, -1)
    song_feature_vectors = song_features.drop(columns=['song_id', 'cluster_id']).values
    similarities = cosine_similarity(user_profile_vector, song_feature_vectors)

    song_features['similarity'] = similarities.flatten()
    recommended_songs = song_features.sort_values(by='similarity', ascending=False).head(top_n)

    song_ids = recommended_songs['song_id'].to_list()
    query = f"""
        SELECT * FROM songs
        WHERE song_id IN ({','.join(f"'{song_id}'" for song_id in song_ids)});
    """
    
    engine = utils.get_engine()
    songs = pd.read_sql(query, engine)

    return songs.to_dict(orient='records')

def get_top_weights(song_ids: list, alpha=0.1):
    if not song_ids:
        return {}
    
    ranks = np.arange(1, len(song_ids) + 1) # Ranks of songs from 1 - len(song_ids) (usually 50)
    weights = np.exp(-alpha * (np.array(ranks) - 1)) # Exponential decay on rank
    weights /= weights.max() # Normalize

    return dict(zip(song_ids, weights)) # Dict using song_id as key and weight as value

def get_rec_weights(song_ids: list, t_liked: pd.Series, l=0.1):
    if not t_liked or len(t_liked) == 0:
        return {}
    
    t_liked = pd.to_datetime(t_liked)

    t_now = pd.Timestamp.now()

    weights = np.exp(-l * (t_now - np.array(t_liked)).dt.total_seconds()) # Exponential decay on time
    weights /= weights.max() # Normalize

    return dict(zip(song_ids, weights)) # Dict using song_id as key and weight as value

def get_saved_weights(song_ids: list, top_weights: dict, base_weight=0.1, boost_factor=0.5):
    saved_weights = {}

    for song_id  in song_ids:
        saved_weights[song_id] = base_weight + top_weights.get(song_id, 0) * boost_factor

    return saved_weights

def get_weighted_vector(feature_vectors: dict, weight_dicts: list):
    combined_weights = {}
    
    for weight_dict in weight_dicts:
        if weight_dict:
            combined_weights.update(weight_dict)

    if not combined_weights:  # If no valid weights, return global average feature vector
        return get_global_avg_vector()  # Implement this to return an average from all songs in the database

    total_weight = sum(combined_weights.values())

    if total_weight == 0:  # Prevent division by zero
        return get_global_avg_vector()

    weighted_sum = sum(np.array(feature_vectors[song]) * weight 
                       for song, weight in combined_weights.items() 
                       if song in feature_vectors)

    return weighted_sum / total_weight

def get_global_avg_vector():
    return [0] * 30

def get_mean_vector(user_id: str):

    query = f'''
        WITH filtered_sf AS (
            SELECT 
                sf.*,
                r.accepted,
                r.created_at
            FROM song_features sf
            LEFT JOIN recommendations r ON sf.song_id = r.song_id AND r.user_id = '{user_id}'
            WHERE sf.song_id IN ({','.join(f"'{song_id}'" for song_id in song_ids)})
            OR r.user_id = '{user_id}'
        )
        SELECT 
            fsf.*, 
            usi.saved, 
            usi.top_song
        FROM filtered_sf fsf
        LEFT JOIN user_song_interactions usi ON fsf.song_id = usi.song_id
        WHERE usi.fsf.song_id IN ({','.join(f"'{song_id}'" for song_id in song_ids)}) OR usi.user_id = '{user_id}';
    '''

    query = f'''
        WITH user_songs AS (
            SELECT 
                COALESCE(usi.song_id, r.song_id) AS song_id,
                usi.top_song,
                usi.saved,
                r.accepted,
                r.created_at
            FROM user_song_interactions usi
            LEFT JOIN recommendations r ON usi.song_id = r.song_id AND r.user_id = '{user_id}'
            WHERE usi.user_id = '{user_id}'
            UNION
            SELECT 
                r.song_id,
                NULL AS top_song,
                NULL AS saved,
                r.accepted,
                r.created_at
            FROM recommendations r
            LEFT JOIN user_song_interactions usi ON r.song_id = usi.song_id AND usi.user_id = '{user_id}'
            WHERE r.user_id = '{user_id}'
        )
        SELECT 
            sf.*,
            us.top_song,
            us.saved,
            us.accepted,
            us.created_at
        FROM song_features sf
        INNER JOIN user_songs us ON sf.song_id = us.song_id;
    '''

    engine = utils.get_engine()
    
    df = pd.read_sql(query, engine)

    top_feature_vectors = df[df['top_song'].notna()].sort_values(by='top_song').drop(columns=['top_song', 'saved', 'accepted', 'created_at'])
    rec_feature_vectors = df[df['accepted'] == 1].sort_values(by='created_at').drop(columns=['top_song', 'saved', 'accepted'])
    saved_feature_vectors = df[df['saved'] == 1].drop(columns=['top_song', 'saved', 'accepted', 'created_at'])

    top_weights = get_top_weights(top_feature_vectors['song_id'].to_list())
    rec_weights = get_rec_weights(rec_feature_vectors['song_id'].to_list(), rec_feature_vectors['created_at'])
    saved_weights = get_saved_weights(saved_feature_vectors['song_id'].to_list(), top_weights)

    matrix = top_feature_vectors.to_numpy() + rec_feature_vectors.drop(columns='created_at').to_numpy() + saved_feature_vectors.to_numpy()
    feature_dict = {row[0]: row[1:].tolist() for row in matrix if row}

    mean_vector = get_weighted_vector(feature_dict, [top_weights, rec_weights, saved_weights])

    return mean_vector

def get_new_user():
    pass

def get_top_songs(user_id: str, top_n=10):

    engine = utils.get_engine()

    query = '''
        WITH user_songs AS (
            SELECT song_id, top_song FROM user_song_interactions
            WHERE user_id = %s and top_song IS NOT NULL
        )
        SELECT 
            s.*,
            us.top_song
        FROM songs s
        INNER JOIN user_songs us ON s.song_id=us.song_id
    '''

    top_songs = pd.read_sql(query, engine, params=(user_id,)).sort_values(by='top_song')
    return top_songs.head(top_n)

if __name__=='__main__':
    TOP_SONGS = False
    REC_WEIGHTS = True
    RECOMMEND = False

    load_dotenv()

    # test_token = os.getenv('USER_TEST_TOKEN')

    if TOP_SONGS:
        engine = utils.get_engine()
        query = '''
            SELECT * FROM users
        '''
        users = pd.read_sql(query, engine)
        user_id = users['user_id'].values[0]
        top_songs = get_top_songs(user_id, top_n=5)
        print(top_songs)
        mean_vector = get_mean_vector(top_songs['song_id'].to_list())
        print(mean_vector)
        print(len(mean_vector))

    if REC_WEIGHTS:
        song_ids = np.arange(1, 11)

        start_timestamp = pd.to_datetime('2024-01-01')
        end_timestamp = pd.to_datetime('2025-01-01')
        
        # Generate random timestamps
        random_timestamps = []
        for _ in range(10):
            # Generate random timestamp between start and end
            random_timestamp = start_timestamp + (end_timestamp - start_timestamp) * np.random.rand()
            random_timestamps.append(random_timestamp)
        
        random_timestamps = pd.Series(random_timestamps)
        rec_weights = get_rec_weights(song_ids, random_timestamps)
        print(song_ids)
        print(random_timestamps)
        print(rec_weights)

    if RECOMMEND:
        engine = utils.get_engine()
        query = '''
            SELECT * FROM users
        '''
        users = pd.read_sql(query, engine)
        user_id = users['user_id'].values[0]

        top_songs = get_top_songs(user_id, top_n=10)
        print('Recs based on:')
        for song in top_songs['title'].to_list():
            print(song)
        mean_vector = get_mean_vector(top_songs['song_id'].to_list())

        engine = utils.get_engine()
        query = '''
            SELECT * FROM song_features
        '''
        song_features = pd.read_sql(query, engine)

        recs = recommend_songs(mean_vector, song_features, top_n=20)
        print('\nRecs:')
        for rec in recs:
            print(rec['title'])