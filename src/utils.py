from sqlalchemy import create_engine, text
from collections import defaultdict
from dotenv import load_dotenv
from requests import post, get
import mysql.connector
import pandas as pd
import numpy as np
import base64
import json
import time
import os

load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
youtube_key = os.getenv('YOUTUBE_KEY')


def safe_api_call(url, headers, params=None, max_retries=3):
    retries = 0
    while retries < max_retries:
        response = get(url, headers=headers, params=params)

        if response.status_code == 200:
            return json.loads(response.content)
        elif response.status_code == 429:
            retry_after = int(response.headers.get("Retry-After", 1))
            retry_after = min(retry_after, 60)
            print(f"Rate limited! Retrying after {retry_after} seconds...")
            time.sleep(retry_after)
            retries += 1
        else:
            print(f"Error {response.status_code}: {response.text}")
            return None

def get_auth_header(token):
    return {'Authorization': 'Bearer ' + token}
    
def connect_to_db():
    db_connection = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )
    return db_connection.cursor()
    
# Returns 7 pandas dataframes to be added to database
# Table 1: Users
# Table 2: Songs
# Table 3: Artists
# Table 4: Artist Genres
# Table 5: Artist <-> Song interactions
# Table 6: User <-> Song Interactions
# Table 7: User <-> Artist Interactions
def get_all_tables(token):
    headers = get_auth_header(token)

    start_time = time.time()
    try:
        user = get_user_info(token)
    except:
        print('Error')
        return -1
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    user_id = user['user_id'].iloc[0]

    result = {
        'users': user,
        'songs': pd.DataFrame(columns=['song_id', 'title', 'img_url', 'preview_url']),
        'artists': pd.DataFrame(columns=['artist_id', 'name']),
        'artist_genres': pd.DataFrame(columns=['artist_id', 'genre']),
        'user_song_interactions': pd.DataFrame(columns=['user_id', 'song_id', 'saved', 'top_song', 'playlist']),
        'user_artist_interactions': pd.DataFrame(columns=['user_id', 'artist_id', 'follows', 'top_artist']),
        'song_artist_interactions': pd.DataFrame(columns=['song_id', 'artist_id'])
    }
    
    start_time = time.time()
    result = get_top(token, headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    start_time = time.time()
    result = get_all_saved_tracks(token, headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    start_time = time.time()
    result = get_followed_artists(token, headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    start_time = time.time()
    result = get_all_playlist_tracks(token, headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    for key in result:
        result[key] = result[key].drop_duplicates()
        
    start_time = time.time()
    result = get_genres(token, headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    return result

def get_user_info(token):
    print('-----------------Getting User Data-----------------')

    url = 'https://api.spotify.com/v1/me'
    headers = get_auth_header(token)
    json_response = safe_api_call(url=url, headers=headers)
    if not json_response:
        raise Exception('Error fetching user')

    return pd.DataFrame.from_dict({
        'user_id': [json_response['id']], 
        'email': [json_response['email']], 
        'profile_img_url': [json_response['images'][0]['url'] if json_response['images'] != [] else '']
    })

def get_top(token, headers, user_id, result):
    print('-----------------Getting Top Tracks/Artists Data-----------------')
    
    url = 'https://api.spotify.com/v1/me/top/tracks'
    params = {
        'limit': 50,
        'offset': 0
    }

    json_response = safe_api_call(url=url, headers=headers, params=params)
    if not json_response:
        return result

    songs_list = []
    artists_list = []
    user_song_interactions_list = []
    song_artist_interaction_list = []
    for item in json_response['items']:
        songs_list.append([item['id'], item['name'], item['album']['images'][0]['url'], '']) # Add song to songs list
        user_song_interactions_list.append([user_id, item['id'], False, True, False]) # Add User Song interaction
            
        for artist in item['artists']:
            artists_list.append([artist['id'], artist['name']]) # Add Artist(s) to artists list
            song_artist_interaction_list.append([item['id'], artist['id']]) # Add Song Artist Interaction

    result['songs'] = pd.concat([result['songs'], pd.DataFrame(data=songs_list, columns=result['songs'].columns)])
    result['artists'] = pd.concat([result['artists'], pd.DataFrame(data=artists_list, columns=result['artists'].columns)])
    result['user_song_interactions'] = pd.concat([result['user_song_interactions'], pd.DataFrame(data=user_song_interactions_list, columns=result['user_song_interactions'].columns)])
    result['song_artist_interactions'] = pd.concat([result['song_artist_interactions'], pd.DataFrame(data=song_artist_interaction_list, columns=result['song_artist_interactions'].columns)])
        
    # Top Artists
    url = 'https://api.spotify.com/v1/me/top/artists'
    params = {
        'limit': 50,
        'offset': 0
    }
    response = get(url=url, headers=headers, params=params)
    json_response = json.loads(response.content)

    artists_list = []
    user_artist_interaction_list = []
    for item in json_response['items']:
        artists_list.append([item['id'], item['name']]) # Add artist to artists list
        user_artist_interaction_list.append([user_id, item['id'], False, True]) # Add User Artist interaction

    result['artists'] = pd.concat([result['artists'], pd.DataFrame(data=artists_list, columns=result['artists'].columns)])
    result['user_artist_interactions'] = pd.concat([result['user_artist_interactions'], pd.DataFrame(data=user_artist_interaction_list, columns=result['user_artist_interactions'].columns)])
    return result

def get_all_saved_tracks(token, headers, user_id, result):
    print('-----------------Getting Saved Tracks-----------------')

    url = 'https://api.spotify.com/v1/me/tracks'
    count = 0

    songs_list = []
    artists_list = []
    user_song_interactions_list = []
    song_artist_interaction_list = []

    while True:
        params = {
            'limit': 50,
            'offset': 50 * count
        }

        json_response = safe_api_call(url=url, headers=headers, params=params)
        if not json_response:
            break

        if 'items' not in json_response or not json_response['items']:
            break

        for item in json_response['items']:
            track = item['track']
            songs_list.append([track['id'], track['name'], track['album']['images'][0]['url'], '']) # Add Song

            # Add User Song Interactions (checking to see if already in table)
            matching_rows = result['user_song_interactions'][
                (result['user_song_interactions']['user_id'] == user_id) & 
                (result['user_song_interactions']['song_id'] == track['id'])
            ]

            if not matching_rows.empty:
                index = matching_rows.index[0]
                result['user_song_interactions'].at[index, 'saved'] = True 
            else:
                user_song_interactions_list.append([user_id, track['id'], True, False, False])
            
            for artist in track['artists']:
                artists_list.append([artist['id'], artist['name']]) # Add Artist(s)
                song_artist_interaction_list.append([track['id'], artist['id']]) # Add Song Artist Interactions

        count += 1
    result['songs'] = pd.concat([result['songs'], pd.DataFrame(data=songs_list, columns=result['songs'].columns)])
    result['artists'] = pd.concat([result['artists'], pd.DataFrame(data=artists_list, columns=result['artists'].columns)])
    result['user_song_interactions'] = pd.concat([result['user_song_interactions'], pd.DataFrame(data=user_song_interactions_list, columns=result['user_song_interactions'].columns)])
    result['song_artist_interactions'] = pd.concat([result['song_artist_interactions'], pd.DataFrame(data=song_artist_interaction_list, columns=result['song_artist_interactions'].columns)])
    return result

def get_all_playlist_tracks(token, headers, user_id, result, print_results=False):
    print('-----------------Getting Playlists-----------------')

    url = f'https://api.spotify.com/v1/users/{user_id}/playlists'
    count = 0
    playlist_count = 0

    while True:
        params = {
            'limit': 50,
            'offset': 50 * count
        }

        if print_results:
            start_time = time.time()

        json_response = safe_api_call(url=url, headers=headers, params=params)

        if print_results:
            end_time = time.time()
            print(f'API request completed in {end_time - start_time} seconds')

        if not json_response or 'items' not in json_response or not json_response['items']:
            break

        for i, item in enumerate(json_response['items']):
            playlist_count += 1

            total_tracks = item['tracks']['total']
            tracks_url = item['tracks']['href']

            track_count = 0

            while track_count < total_tracks:
                params = {
                    'limit': 100,
                    'offset': track_count
                }
                
                if print_results:
                    start_time = time.time()

                json_tracks_response = safe_api_call(url=tracks_url, headers=headers, params=params)

                if print_results:
                    end_time = time.time()
                    print(f'API request completed in {end_time - start_time} seconds')
                    
                if not json_tracks_response or 'items' not in json_tracks_response or not json_tracks_response['items']:
                    break

                num_fetched = len(json_tracks_response['items'])
                track_count += num_fetched

                songs_list = []
                artists_list = []
                user_song_interactions_list = []
                song_artist_interactions_list = []

                for item in json_tracks_response['items']:
                    if track_count >= total_tracks:
                        break
                            
                    track = item['track']
                    if track and track.get('id') and track.get('name') and track.get('artists'):
                        
                        songs_list.append([track['id'], track['name'], track['album']['images'][0]['url'], '']) # Add song to df

                        # Add User Song Interactions (checking to see if already in table)
                        matching_rows = result['user_song_interactions'][
                            (result['user_song_interactions']['user_id'] == user_id) & 
                            (result['user_song_interactions']['song_id'] == track['id'])
                        ]

                        if not matching_rows.empty:
                            index = matching_rows.index[0]
                            result['user_song_interactions'].at[index, 'playlist'] = True 
                        else:
                            user_song_interactions_list.append([user_id, track['id'], False, False, True])

                        # Add artists to table
                        for artist in track['artists']:
                            artists_list.append([artist['id'], artist['name']]) # Add Artist(s)
                            song_artist_interactions_list.append([track['id'], artist['id']]) # Add Song Artist Interactions

                result['songs'] = pd.concat([result['songs'], pd.DataFrame(data=songs_list, columns=result['songs'].columns)])
                result['artists'] = pd.concat([result['artists'], pd.DataFrame(data=artists_list, columns=result['artists'].columns)])
                result['user_song_interactions'] = pd.concat([result['user_song_interactions'], pd.DataFrame(data=user_song_interactions_list, columns=result['user_song_interactions'].columns)])
                result['song_artist_interactions'] = pd.concat([result['song_artist_interactions'], pd.DataFrame(data=song_artist_interactions_list, columns=result['song_artist_interactions'].columns)])    

        if(playlist_count >= json_response['total']):
            break

        count += 1
    return result

def get_followed_artists(token, headers, user_id, result):
    print('-----------------Getting Followed Artists-----------------')

    url = 'https://api.spotify.com/v1/me/following?type=artist&limit=50'

    user_artist_interactions_list = []
    while True:
        response = get(url=url, headers=headers)

        json_response = json.loads(response.content)

        if 'artists' not in json_response or not json_response['artists']:
            break

        for artist in json_response['artists']:
            if 'item' not in artist or not artist['items']:
                break
            for item in artist['items']:
                matching_rows = result['user_artist_interactions'][
                    (result['user_artist_interactions']['user_id'] == user_id) & 
                    (result['user_artist_interactions']['artist_id'] == item['id'])
                ]

                if not matching_rows.empty:
                    index = matching_rows.index[0]
                    result['user_artist_interactions'].at[index, 'follows'] = True 
                else:
                    user_artist_interactions_list.append([user_id, item['id'], True, False])
                after = item['id']

        if json_response['artists']['next']:
            url = json_response['artists']['next']
        else:
            break

    result['user_artist_interactions'] = pd.concat([result['user_artist_interactions'], pd.DataFrame(data=user_artist_interactions_list, columns=result['user_artist_interactions'].columns)])
    return result

def get_genres(token, headers, user_id, result):
    print('-----------------Getting Genre Data-----------------')

    url = 'https://api.spotify.com/v1/artists'
    artists = result['artists']
    artist_ids = artists['artist_id'].tolist()
    artist_genres = result['artist_genres']
    end = 50

    artist_genres_list = []
    while end < len(artist_ids) or abs(len(artist_ids) - end) < 50:
        params = {
            'ids': ','.join(artist_ids[end-50:end])
        }
        response = get(url=url, headers=headers, params=params)
        json_response = json.loads(response.content)

        for artist in json_response['artists']:
            artist_id = artist['id']
            if artist.get('genres'):
                for genre in artist['genres']:
                    artist_genres_list.append([artist_id, genre])
        end += 50
    result['artist_genres'] = pd.concat([result['artist_genres'], pd.DataFrame(data=artist_genres_list, columns=result['artist_genres'].columns)])
    return result

def add_df_to_db(dfs):
    if 'users' in dfs and len(dfs['users']) > 0:
        users = dfs['users']
        connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        user_data = users.to_dict(orient='records')

        columns = ', '.join(users.columns)
        update_values = ', '.join([f'{col}=VALUES({col})' for col in users.columns])

        query = text(f'''
            INSERT INTO users ({columns})
            VALUES ({', '.join([f':{col}' for col in users.columns])})
            ON DUPLICATE KEY UPDATE 
                email = VALUES(email), 
                profile_img_url = VALUES(profile_img_url), 
                last_updated = NOW()
        ''')

        with engine.connect() as conn:
            for row in user_data:
                conn.execute(query, row)
            conn.commit()

        print(f'USER data inserted successfully')

    if 'songs' in dfs and len(dfs['songs']) > 0:
        songs = dfs['songs']
        connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        songs_data = songs.to_dict(orient='records')
        # print(songs_data)

        columns = ', '.join(songs.columns)
        update_values = ', '.join([f'{col}=VALUES({col})' for col in songs.columns])

        query = f'''
            INSERT INTO songs ({columns})
            VALUES ({', '.join([f':{col}' for col in songs.columns])})
            ON DUPLICATE KEY UPDATE {update_values}
        '''

        with engine.connect() as conn:
            conn.execute(text(query), songs_data)
            conn.commit()

        print(f'SONG data inserted successfully')

    if 'artists' in dfs and len(dfs['artists']) > 0:
        artists = dfs['artists']
        connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        artists_data = artists.to_dict(orient='records')

        columns = ', '.join(artists.columns)
        update_values = ', '.join([f'{col}=VALUES({col})' for col in artists.columns])

        query = f'''
            INSERT INTO artists ({columns})
            VALUES ({', '.join([f':{col}' for col in artists.columns])})
            ON DUPLICATE KEY UPDATE {update_values}
        '''

        with engine.connect() as conn:
            conn.execute(text(query), artists_data)
            conn.commit()

        print(f'ARTIST data inserted successfully')

    if 'user_song_interactions' in dfs and len(dfs['user_song_interactions']) > 0:
        user_song_interactions = dfs['user_song_interactions']
        connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        user_song_interactions_data = user_song_interactions.to_dict(orient='records')

        columns = ', '.join(user_song_interactions.columns)
        update_values = ', '.join([f'{col}=VALUES({col})' for col in user_song_interactions.columns if col != 'user_id' and col != 'song_id'])

        query = f'''
            INSERT INTO user_song_interactions ({columns})
            VALUES ({', '.join([f':{col}' for col in user_song_interactions.columns])})
            ON DUPLICATE KEY UPDATE {update_values}
        '''

        with engine.connect() as conn:
            conn.execute(text(query), user_song_interactions_data)
            conn.commit()

        print(f'USER <-> SONG data inserted successfully')

    if 'user_artist_interactions' in dfs and len(dfs['user_artist_interactions']) > 0:
        user_artist_interactions = dfs['user_artist_interactions']
        connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        user_artist_interactions_data = user_artist_interactions.to_dict(orient='records')

        columns = ', '.join(user_artist_interactions.columns)
        update_values = ', '.join([f'{col}=VALUES({col})' for col in user_artist_interactions.columns if col != 'user_id' and col != 'artist_id'])

        query = f'''
            INSERT INTO user_artist_interactions ({columns})
            VALUES ({', '.join([f':{col}' for col in user_artist_interactions.columns])})
            ON DUPLICATE KEY UPDATE {update_values}
        '''

        with engine.connect() as conn:
            conn.execute(text(query), user_artist_interactions_data)
            conn.commit()

        print(f'USER <-> SONG data inserted successfully')

    if 'song_artist_interactions' in dfs and len(dfs['song_artist_interactions']) > 0:
        song_artist_interactions = dfs['song_artist_interactions']
        connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        song_artist_interactions_data = song_artist_interactions.to_dict(orient='records')

        columns = ', '.join(song_artist_interactions.columns)

        query = f'''
            INSERT INTO song_artist_interactions ({columns})
            VALUES ({', '.join([f':{col}' for col in song_artist_interactions.columns])})
            ON DUPLICATE KEY UPDATE song_id = song_id
        '''

        with engine.connect() as conn:
            conn.execute(text(query), song_artist_interactions_data)
            conn.commit()

        print(f'SONG <-> ARTIST data inserted successfully')

    if 'artist_genres' in dfs and len(dfs['artist_genres']) > 0:
        artist_genres = dfs['artist_genres']
        connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        artist_genres_data = artist_genres.to_dict(orient='records')

        columns = ', '.join(artist_genres.columns)

        query = f'''
            INSERT INTO artist_genres ({columns})
            VALUES ({', '.join([f':{col}' for col in artist_genres.columns])})
            ON DUPLICATE KEY UPDATE artist_id = artist_id
        '''

        with engine.connect() as conn:
            conn.execute(text(query), artist_genres_data)
            conn.commit()

        print(f'ARTIST <-> GENRE data inserted successfully')

def get_preview(song_name, artist_name):
    try:
        url = 'https://www.googleapis.com/youtube/v3/search'
        query = f'{song_name} {artist_name} official audio'
        params = {
            'part': 'snippet',
            'q': query,
            'key': youtube_key,
            'maxResults': 1,
            'type': 'video'
        }

        response = get(url=url, params=params)
        json_response = response.json()
        print(json_response)

        if 'items' in json_response and json_response['items']:
            video_id = json_response['items'][0]['id']['videoId']
            return f'https://www.youtube.com/watch?v={video_id}'
    except Exception as e:
        print(f'YouTube API error: {e}')

    return ''

def get_artist_name(songs, artists, song_artist_interactions, print_results=False):
    song_ids = songs['song_id'].to_list()
    result = {}
    for song_id in song_ids:
        song_title = songs[songs['song_id'] == song_id]['title'].values[0]
        artist_id = song_artist_interactions[song_artist_interactions['song_id'] == song_id]['artist_id'].values[0]
        artist_name = artists[artists['artist_id'] == artist_id]['name'].values[0]
        result[song_id] = (song_title, artist_name)
        if print_results:
            print(f'{song_title} <-> {artist_name}')
            print(f'{song_id} <-> {artist_id}\n')
    return result

def put_all_previews(save=True):

    result = load_tables(['songs', 'artists', 'song_artist_interactions'])
    songs, artists, song_artist_interactions = result['songs'], result['artists'], result['song_artist_interactions']

    print('\nRetrieving preview urls...\n')

    start_time = time.time()

    song_ids = get_artist_name(songs, artists, song_artist_interactions)

    for song_id in song_ids:
        song, artist = song_ids[song_id]

        preview_value = songs[songs['song_id'] == song_id]['preview_url'].iloc[0]

        if preview_value == '' or pd.isna(preview_value):
            preview = get_preview(song, artist)
            songs.loc[songs['song_id'] == song_id, 'preview_url'] = preview
            print(f'{song} by {artist}: {preview}')

    end_time = time.time()
    print(f'\nAll preview retrieved in {end_time - start_time} seconds\n')

    print(songs['preview_url'])
    print(f'Number of missing Previews: {songs['preview_url'].isna().sum() + songs['preview_url'].eq('').sum()}')

    if save:
        add_df_to_db({'songs': songs})
    

def load_tables(tables):
    print('Loading Data...\n\n\n')
    connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_string)

    result = {}

    for table in tables:
        query = f'SELECT * FROM {table}'
        df = pd.read_sql(query, engine)
        result[table] = df
    print('Data successfully loaded')
    return result

if __name__=='__main__':
    USER = False
    GENRE = False
    TABLES = False
    CONNECT = False
    YOUTUBE = False
    TOP = False
    SAVED = False
    PLAYLIST = False
    FOLLOWS = False
    SAVE_DATA = False
    SAVE_PREVIEWS = False
    LOAD_DATA = True
    ARTIST_NAMES = False
    

    test_token = os.getenv('USER_TEST_TOKEN')
    
    if USER:
        start_time = time.time()
        result = get_user_info(test_token)
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')
        
        print(result)

    if GENRE:
        headers = get_auth_header(test_token)
        result = {
            'users': get_user_info(test_token),
            'songs': pd.DataFrame(columns=['song_id', 'title', 'img_url', 'preview_url']),
            'artists': pd.DataFrame(columns=['artist_id', 'name']),
            'artist_genres': pd.DataFrame(columns=['artist_id', 'genre']),
            'user_song_interactions': pd.DataFrame(columns=['user_id', 'song_id', 'saved', 'top_song', 'playlist']),
            'user_artist_interactions': pd.DataFrame(columns=['user_id', 'artist_id', 'follows', 'top_artist']),
            'song_artist_interactions': pd.DataFrame(columns=['song_id', 'artist_id'])
        }
        user_id = result['users']['user_id'].iloc[0]

        result = get_top(test_token, headers, user_id, result)
        for key in result:
            result[key] = result[key].drop_duplicates()
        start_time = time.time()
        result = get_genres(test_token, headers, user_id, result)
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')

        print(f'Artist Genres:\n{len(result['artist_genres'])} rows\n{result['artist_genres']['artist_id'].nunique()} unique artist ID\'s\n{result['artist_genres'].head(3)}')

    if TABLES:
        start_time = time.time()
        result = get_all_tables(test_token)
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')

        for key in result:
            print(f'{key}: {len(result[key])} rows')
            print(result[key].head(3))

    if CONNECT:
        print('Connecting to database...')
        cursor = connect_to_db()
        if cursor:
            print('Connected to database successfully')

    if YOUTUBE:
        start_time = time.time()
        link = get_preview('Take Me Out', 'Franz Ferdinand')
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')

        print(link)

    if TOP:
        headers = get_auth_header(test_token)
        result = {
            'users': get_user_info(test_token),
            'songs': pd.DataFrame(columns=['song_id', 'title', 'img_url', 'preview_url']),
            'artists': pd.DataFrame(columns=['artist_id', 'name']),
            'artist_genres': pd.DataFrame(columns=['artist_id', 'genre']),
            'user_song_interactions': pd.DataFrame(columns=['user_id', 'song_id', 'saved', 'top_song', 'playlist']),
            'user_artist_interactions': pd.DataFrame(columns=['user_id', 'artist_id', 'follows', 'top_artist']),
            'song_artist_interactions': pd.DataFrame(columns=['song_id', 'artist_id'])
        }
        user_id = result['users']['user_id'].iloc[0]
        start_time = time.time()
        result = get_top(test_token, headers, user_id, result)
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')

        for key in result:
            df = result[key]
            print(f'{key}: {len(df)} rows\n{df.head(3)}\n')

    if SAVED:
        headers = get_auth_header(test_token)
        result = {
            'users': get_user_info(test_token),
            'songs': pd.DataFrame(columns=['song_id', 'title', 'img_url', 'preview_url']),
            'artists': pd.DataFrame(columns=['artist_id', 'name']),
            'artist_genres': pd.DataFrame(columns=['artist_id', 'genre']),
            'user_song_interactions': pd.DataFrame(columns=['user_id', 'song_id', 'saved', 'top_song', 'playlist']),
            'user_artist_interactions': pd.DataFrame(columns=['user_id', 'artist_id', 'follows', 'top_artist']),
            'song_artist_interactions': pd.DataFrame(columns=['song_id', 'artist_id'])
        }
        user_id = result['users']['user_id'].iloc[0]
        start_time = time.time()
        result = get_all_saved_tracks(test_token, headers, user_id, result)
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')

        for key in result:
            df = result[key]
            print(f'{key}: {len(df)} rows\n{df.head(3)}\n')

    if PLAYLIST:
        headers = get_auth_header(test_token)
        result = {
            'users': get_user_info(test_token),
            'songs': pd.DataFrame(columns=['song_id', 'title', 'img_url', 'preview_url']),
            'artists': pd.DataFrame(columns=['artist_id', 'name']),
            'artist_genres': pd.DataFrame(columns=['artist_id', 'genre']),
            'user_song_interactions': pd.DataFrame(columns=['user_id', 'song_id', 'saved', 'top_song', 'playlist']),
            'user_artist_interactions': pd.DataFrame(columns=['user_id', 'artist_id', 'follows', 'top_artist']),
            'song_artist_interactions': pd.DataFrame(columns=['song_id', 'artist_id'])
        }
        user_id = result['users']['user_id'].iloc[0]

        start_time = time.time()
        result = get_all_playlist_tracks(test_token, headers, user_id, result, print_results=True)
        end_time = time.time()

        print(f'Completed in {end_time - start_time} seconds')
        
        for key in result:
            df = result[key]
            print(f'{key}: {len(df)} rows\n{df.head(3)}\n')

        while True:
            save = input('Save data to database? [y/n]')
            if save == 'y':
                add_df_to_db(result)
                break
            elif save == 'n':
                break
            else:
                print('Invalid input')


    if FOLLOWS:
        headers = get_auth_header(test_token)
        result = {
            'users': get_user_info(test_token),
            'songs': pd.DataFrame(columns=['song_id', 'title', 'img_url', 'preview_url']),
            'artists': pd.DataFrame(columns=['artist_id', 'name']),
            'artist_genres': pd.DataFrame(columns=['artist_id', 'genre']),
            'user_song_interactions': pd.DataFrame(columns=['user_id', 'song_id', 'saved', 'top_song', 'playlist']),
            'user_artist_interactions': pd.DataFrame(columns=['user_id', 'artist_id', 'follows', 'top_artist']),
            'song_artist_interactions': pd.DataFrame(columns=['song_id', 'artist_id'])
        }
        user_id = result['users']['user_id'].iloc[0]
        start_time = time.time()
        result = get_followed_artists(test_token, headers, user_id, result)
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')
        
        for key in result:
            df = result[key]
            print(f'{key}: {len(df)} rows\n{df.head(3)}\n')
    if SAVE_DATA:

        result = get_all_tables(test_token)
        add_df_to_db(result)

    if SAVE_PREVIEWS:
        
        put_all_previews()

    if LOAD_DATA:

        tables = ['users', 'songs', 'artists', 'artist_genres', 'user_song_interactions', 'user_artist_interactions', 'song_artist_interactions']
        result = load_tables(tables)

        for key in result:
            df = result[key]
            print('=========================================================================================================================================================')
            print(f'{key}:\n{len(df)} row(s)\n{df[df.columns[0]].nunique()} unique {df.columns[0]} value(s)\n\n{df.head(5)}')
            print('=========================================================================================================================================================\n')

    if ARTIST_NAMES:
        result = load_tables(['songs', 'artists', 'song_artist_interactions'])
        songs, artists, song_artist_interactions = result['songs'], result['artists'], result['song_artist_interactions']
        song_titles, artist_names = get_artist_name(songs, artists, song_artist_interactions)
        print(f'Songs: {len(songs)} Artists: {len(artist_names)}')
