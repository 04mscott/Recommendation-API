import utils
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sqlalchemy import text
import numpy as np
from dotenv import load_dotenv
import random


def check_user(user_id: str):
    try:
        query = text('''
            SELECT * FROM users
            WHERE user_id = :user_id;
        ''')

        engine = utils.get_engine()
        with engine.connect() as conn:
            user_data = pd.read_sql(query, conn, params={'user_id': user_id})
        return len(user_data) > 0
    except Exception as e:
        print(f'Error checking user_id: {e}')
        return False

def recommend_songs(user_id: str, top_n: int = 20, noise_factor: int = 0.1):

    song_ids = utils.get_all_user_songs(user_id)

    query = text('''
        SELECT * FROM song_features
        WHERE song_id NOT IN :song_ids;
    ''')

    engine = utils.get_engine()
    with engine.connect() as conn:
        song_features = pd.read_sql(query, conn, params={'song_ids': tuple(song_ids)})

    user_profile_vector = get_mean_vector(user_id)

    user_profile_vector = np.array(user_profile_vector).reshape(1, -1)
    song_feature_vectors = song_features.drop(columns='song_id').values
    similarities = cosine_similarity(user_profile_vector, song_feature_vectors)

    song_features['similarity'] = similarities.flatten()
    recommended_songs = song_features.sort_values(by='similarity', ascending=False).head(top_n)

    num_noisy_songs = max(1, int(top_n * noise_factor))  # How many noisy songs to add (1 or 2)
    
    # Get less similar songs (e.g., bottom 10% of similarities)
    less_similar_songs = song_features.sort_values(by='similarity').head(num_noisy_songs)

    # Replace a few of the recommended songs with less similar songs
    noisy_song_ids = less_similar_songs['song_id'].to_list()
    noisy_song_indices = random.sample(range(top_n), num_noisy_songs)
    
    # Replace the songs in the noisy indices
    for i, noisy_index in enumerate(noisy_song_indices):
        recommended_songs.iloc[noisy_index, recommended_songs.columns.get_loc('song_id')] = noisy_song_ids[i]


    song_ids = recommended_songs['song_id'].to_list()

    query = text("""
        SELECT 
            s.song_id,
            s.title,
            GROUP_CONCAT(a.name ORDER BY a.name SEPARATOR ', ') AS artists,
            s.img_url,
            s.preview_url
        FROM songs s
        LEFT JOIN song_artist_interactions sai ON s.song_id = sai.song_id
        LEFT JOIN artists a ON sai.artist_id = a.artist_id
        WHERE s.song_id IN :song_ids
        GROUP BY s.song_id, s.title, s.img_url, s.preview_url;
    """)
    
    engine = utils.get_engine()
    with engine.connect() as conn:
        songs = pd.read_sql(query, conn, params={'song_ids': tuple(song_ids)})

    return songs.to_dict(orient='records')

def get_top_weights(song_ids: list, alpha: int = 0.05):
    if len(song_ids) == 0:
        return {}
    
    ranks = np.arange(1, len(song_ids) + 1) # Ranks of songs from 1 - len(song_ids) (usually 50)
    weights = np.exp(-alpha * (ranks - 1)) # Exponential decay on rank
    if weights.max() > 0:
        weights /= weights.max() # Normalize

    return dict(zip(song_ids, weights)) # Dict using song_id as key and weight as value

def get_rec_weights(song_ids: list, t_liked: pd.Series, alpha: int = 0.002):
    if len(t_liked) == 0:
        return {}
    
    t_now = pd.Timestamp.now()
    t_liked = pd.to_datetime(t_liked)

    t_diff = (t_now - t_liked).dt.total_seconds() / 86400
    weights = np.exp(-alpha * t_diff) # Exponential decay on time
    
    if weights.max() > 0:
        weights /= weights.max() # Normalize

    return dict(zip(song_ids, weights)) # Dict using song_id as key and weight as value

def get_saved_weights(song_ids: list, top_weights: dict, base_weight: int = 0.1, boost_factor: int = 0.5):
    weights = [
        base_weight + top_weights.get(song_id, 0) * boost_factor
        for song_id in song_ids
    ]

    max_weight = max(weights, default=1)
    return {song_id: weight / max_weight for song_id, weight in zip(song_ids, weights)}  # Return normalized weights

def get_weighted_mean_vector(features_dict: dict):
    # Default values
    means = {
        'top_songs': np.zeros(30),
        'rec_songs': np.zeros(30),
        'saved_songs': np.zeros(30)
    }
    # Default weights
    default_weights = {
        'top_songs': 0.5,
        'rec_songs': 0.35,
        'saved_songs': 0.15 
    }

    for key in features_dict:
        features = features_dict[key]['features']
        weights = features_dict[key]['weights']

        # Set weight of missing features to 0 for final mean
        if len(weights) == 0 or len(features) == 0 or max(weights.values()) <= 0 :
            default_weights[key] = 0
            continue
        
        total_weight = sum(weights.values())
        if total_weight == 0:
            continue

        weighted_sum = np.zeros(30)
        for song_id, feature_vector in features.items():
            if song_id in weights:
                weighted_sum += np.array(feature_vector, dtype=np.float64) * weights[song_id]

        means[key] = weighted_sum / total_weight
        
    total_weight = sum(default_weights.values())

    if total_weight > 0:
        default_weights = {key: weight / total_weight for key, weight in default_weights.items()}
       
    weighted_mean = sum(means[key] * default_weights[key] for key in means)

    if np.all(weighted_mean == 0) or total_weight == 0:
        return get_global_avg_vector()

    return weighted_mean

def get_global_avg_vector():
    query = '''
        SELECT * FROM song_features;
    '''

    engine = utils.get_engine()

    sf = pd.read_sql(query, engine)
    
    features = sf.drop(columns='song_id').to_numpy()

    return np.mean(features, axis=0)

def get_mean_vector(user_id: str):

    query = text('''
        SELECT 
            sf.*,
            usi.top_song,
            usi.saved,
            usi.recommended,
            usi.last_updated
        FROM song_features sf
        INNER JOIN user_song_interactions usi ON sf.song_id = usi.song_id
        WHERE usi.user_id = :user_id;
    ''')

    engine = utils.get_engine()
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={'user_id': user_id})

    top_feature_vectors = df[df['top_song'].notna()].sort_values(by='top_song').drop(columns=['top_song', 'saved', 'recommended', 'last_updated'])
    rec_feature_vectors = df[df['recommended'] == 'liked'].sort_values(by='last_updated').drop(columns=['top_song', 'saved', 'recommended'])
    saved_feature_vectors = df[df['saved'] == 1].drop(columns=['top_song', 'saved', 'recommended', 'last_updated'])

    top_weights = get_top_weights(top_feature_vectors['song_id'].to_list())
    rec_weights = get_rec_weights(rec_feature_vectors['song_id'].to_list(), rec_feature_vectors['last_updated'])
    saved_weights = get_saved_weights(saved_feature_vectors['song_id'].to_list(), top_weights)

    mean_vector = get_weighted_mean_vector({
        'top_songs': {
            'features': {row[0]: row[1:] for row in top_feature_vectors.to_numpy()},
            'weights': top_weights
        },
        'rec_songs': {
            'features': {row[0]: row[1:] for row in rec_feature_vectors.drop(columns='last_updated').to_numpy()},
            'weights': rec_weights
        },
        'saved_songs': {
            'features': {row[0]: row[1:] for row in saved_feature_vectors.to_numpy()},
            'weights': saved_weights
        }
    })

    return mean_vector

def get_new_user():
    pass

def get_top_songs(user_id: str, top_n: int = 10):

    engine = utils.get_engine()

    query = text('''
        SELECT 
            s.*,
            usi.top_song
        FROM songs s
        INNER JOIN user_song_interactions usi ON s.song_id = usi.song_id
        WHERE usi.top_song IS NOT NULL AND user_id = :user_id;
    ''')

    with engine.connect() as conn:
        top_songs = pd.read_sql(query, conn, params={'user_id': user_id}).sort_values(by='top_song')
    return top_songs.head(top_n)

if __name__=='__main__':
    GET_USER = True
    TOP_SONGS = False
    TOP_WEIGHTS = False
    REC_WEIGHTS = False
    SAVED_WEIGHTS = False
    MEAN_VECTOR = False
    RECOMMEND = False

    load_dotenv()

    # test_token = os.getenv('USER_TEST_TOKEN')

    if GET_USER:
        engine = utils.get_engine()
        query = '''
            SELECT * FROM users
        '''
        users = pd.read_sql(query, engine)
        print(users[['user_id', 'email']])
    
    if TOP_SONGS:
        engine = utils.get_engine()
        query = '''
            SELECT * FROM users
        '''
        users = pd.read_sql(query, engine)
        user_id = users['user_id'].values[0]
        top_songs = get_top_songs(user_id, top_n=5)
        print(top_songs)

    if TOP_WEIGHTS:
        song_ids = np.arange(1, 51)
        top_weights = get_top_weights(song_ids)
        print(top_weights)

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
        # print(song_ids)
        # print(random_timestamps)
        print(rec_weights)

    if SAVED_WEIGHTS:
        song_ids = np.arange(1, 11)
        top_weights = get_top_weights(song_ids)
        saved_weights = get_saved_weights(song_ids, top_weights)
        print(saved_weights)

        song_ids = np.arange(6, 16)
        saved_weights = get_saved_weights(song_ids, top_weights)
        print(saved_weights)

    if MEAN_VECTOR:
        engine = utils.get_engine()
        query = '''
            SELECT * FROM users
        '''
        users = pd.read_sql(query, engine)
        user_id = users['user_id'].values[0]

        mean_vector = get_mean_vector(user_id)
        print(mean_vector)

        mean_vector = get_mean_vector('fake_user')
        print(mean_vector)

    if RECOMMEND:
        engine = utils.get_engine()
        query = '''
            SELECT * FROM users
        '''
        users = pd.read_sql(query, engine)
        user_id = users['user_id'].values[0]

        recs = recommend_songs(user_id, top_n=20)
        for rec in recs:
            print(rec['title'])
        
