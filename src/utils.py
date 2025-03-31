from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sqlalchemy import create_engine, text
from pydub.utils import mediainfo
from dotenv import load_dotenv
from requests import get
import mysql.connector
import librosa as lr
import pandas as pd
import numpy as np
import subprocess
import audioread
import yt_dlp
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

CURR_DIR = current_directory_os = os.getcwd()

def safe_api_call(url: str, headers: dict, params=None, max_retries=3):
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

def get_auth_header(token: str):
    return {'Authorization': 'Bearer ' + token}
    
def connect_to_db():
    conn = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name
    )
    return conn
    
def get_engine():
    connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    return create_engine(connection_string)

# Returns 7 pandas dataframes to be added to database
# Table 1: Users
# Table 2: Songs
# Table 3: Artists
# Table 4: Artist Genres
# Table 5: Artist <-> Song interactions
# Table 6: User <-> Song Interactions
# Table 7: User <-> Artist Interactions
def get_init_tables(token: str):
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
    result = get_top(headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    start_time = time.time()
    result = get_all_saved_tracks(headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    start_time = time.time()
    result = get_followed_artists(headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    start_time = time.time()
    result = get_all_playlist_tracks(headers, user_id, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    for key in result:
        result[key] = result[key].drop_duplicates()

    start_time = time.time()
    result = get_previews(result, download=False)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')
        
    start_time = time.time()
    result = get_genres(headers, result)
    end_time = time.time()
    print(f'Completed in {end_time - start_time} seconds\n')

    return result

def get_user_info(token: str):
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

def get_top(headers: dict, user_id: str, result: dict, limit=50, top_artists=True):
    print('-----------------Getting Top Tracks/Artists Data-----------------')
    
    url = 'https://api.spotify.com/v1/me/top/tracks'
    params = {
        'limit': limit,
        'offset': 0
    }

    json_response = safe_api_call(url=url, headers=headers, params=params)
    if not json_response:
        return result

    songs_list = []
    artists_list = []
    user_song_interactions_list = []
    song_artist_interaction_list = []
    for i, item in enumerate(json_response['items']):
        songs_list.append([item['id'], item['name'], item['album']['images'][0]['url'], '']) # Add song to songs list
        user_song_interactions_list.append([user_id, item['id'], False, i, False]) # Add User Song interaction
            
        for artist in item['artists']:
            artists_list.append([artist['id'], artist['name']]) # Add Artist(s) to artists list
            song_artist_interaction_list.append([item['id'], artist['id']]) # Add Song Artist Interaction

    result['songs'] = pd.concat([result['songs'], pd.DataFrame(data=songs_list, columns=result['songs'].columns)])
    result['artists'] = pd.concat([result['artists'], pd.DataFrame(data=artists_list, columns=result['artists'].columns)])
    result['user_song_interactions'] = pd.concat([result['user_song_interactions'], pd.DataFrame(data=user_song_interactions_list, columns=result['user_song_interactions'].columns)])
    result['song_artist_interactions'] = pd.concat([result['song_artist_interactions'], pd.DataFrame(data=song_artist_interaction_list, columns=result['song_artist_interactions'].columns)])
        
    # Top Artists
    if top_artists:
        url = 'https://api.spotify.com/v1/me/top/artists'
        params = {
            'limit': limit,
            'offset': 0
        }
        response = get(url=url, headers=headers, params=params)
        json_response = json.loads(response.content)

        artists_list = []
        user_artist_interaction_list = []
        for i, item in enumerate(json_response['items']):
            artists_list.append([item['id'], item['name']]) # Add artist to artists list
            user_artist_interaction_list.append([user_id, item['id'], False, i]) # Add User Artist interaction

        result['artists'] = pd.concat([result['artists'], pd.DataFrame(data=artists_list, columns=result['artists'].columns)])
        result['user_artist_interactions'] = pd.concat([result['user_artist_interactions'], pd.DataFrame(data=user_artist_interaction_list, columns=result['user_artist_interactions'].columns)])
    return result

def get_all_saved_tracks(headers: dict, user_id: str, result: dict):
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
                user_song_interactions_list.append([user_id, track['id'], True, None, False])
            
            for artist in track['artists']:
                artists_list.append([artist['id'], artist['name']]) # Add Artist(s)
                song_artist_interaction_list.append([track['id'], artist['id']]) # Add Song Artist Interactions

        count += 1
    result['songs'] = pd.concat([result['songs'], pd.DataFrame(data=songs_list, columns=result['songs'].columns)])
    result['artists'] = pd.concat([result['artists'], pd.DataFrame(data=artists_list, columns=result['artists'].columns)])
    result['user_song_interactions'] = pd.concat([result['user_song_interactions'], pd.DataFrame(data=user_song_interactions_list, columns=result['user_song_interactions'].columns)])
    result['song_artist_interactions'] = pd.concat([result['song_artist_interactions'], pd.DataFrame(data=song_artist_interaction_list, columns=result['song_artist_interactions'].columns)])
    return result

def get_all_playlist_tracks(headers: dict, user_id: str, result: dict, print_results=False):
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
                            user_song_interactions_list.append([user_id, track['id'], False, None, True])

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

def get_followed_artists(headers: dict, user_id: str, result: dict):
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
                    user_artist_interactions_list.append([user_id, item['id'], True, None])
                after = item['id']

        if json_response['artists']['next']:
            url = json_response['artists']['next']
        else:
            break

    result['user_artist_interactions'] = pd.concat([result['user_artist_interactions'], pd.DataFrame(data=user_artist_interactions_list, columns=result['user_artist_interactions'].columns)])
    return result

def get_genres(headers: dict, result: dict):
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

def add_df_to_db(dfs: dict):
    connection_string = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_string)
    if 'users' in dfs and len(dfs['users']) > 0:
        users = dfs['users']
        users_data = users.to_dict(orient='records')

        columns = ', '.join(users.columns)
        update_values = ', '.join([f'{col}=VALUES({col})' for col in users.columns])

        query = f'''
            INSERT INTO users ({columns})
            VALUES ({', '.join([f':{col}' for col in users.columns])})
            ON DUPLICATE KEY UPDATE {update_values}
        '''

        with engine.connect() as conn:
            conn.execute(text(query), users_data)
            conn.commit()

        print(f'USER data inserted successfully')

    if 'songs' in dfs and len(dfs['songs']) > 0:
        songs = dfs['songs']
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

    if 'song_features' in dfs and len(dfs['song_features']) > 0:
        song_features = dfs['song_features']
        song_features_data = song_features.to_dict(orient='records')

        columns = ', '.join(song_features.columns)
        update_values = ', '.join([f'{col}=VALUES({col})' for col in song_features.columns if col != 'song_id'])

        query = f'''
            INSERT INTO song_features ({columns})
            VALUES ({', '.join([f':{col}' for col in song_features.columns])})
            ON DUPLICATE KEY UPDATE {update_values}
        '''

        with engine.connect() as conn:
            conn.execute(text(query), song_features_data)
            conn.commit()

        print(f'SONG FEATURES data inserted successfully')

    if 'artists' in dfs and len(dfs['artists']) > 0:
        artists = dfs['artists']
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

def get_artist_name(song_ids: list, result: dict):

    if not song_ids:
        print('song_ids empty')
        return {}
    
    s_df = result['songs']
    a_df = result['artists']
    sai_df = result['song_artist_interactions']
    
    song_artists = {}
    for song_id in song_ids:
        song_row = s_df[s_df['song_id'] == song_id]
        if not song_row.empty:
            song_title = song_row.iloc[0]['title']
        else:
            print(f"Song ID {song_id} not found in songs data")
            continue
        artist_ids = sai_df[sai_df['song_id'] == song_id]['artist_id'].tolist()
        artist_names = []
        for artist_id in artist_ids:
            artist_row = a_df[a_df['artist_id'] == artist_id]
            if not artist_row.empty:
                artist_names.append(artist_row.iloc[0]['name'])
            else:
                print(f"Artist ID {artist_id} not found in artists data")
        if song_id in song_artists:
            song_artists[song_id][1].extend(artist_names)
        else:
            song_artists[song_id] = (song_title, artist_names)

    return song_artists



    return song_artists

def fetch_urls_for_batch(batch, result):
    
    names = get_artist_name(batch, result)

    def fetch_url(song_id):
        if song_id in names:
            song_title, artist_names = names[song_id]
            artist_name = artist_names[0]
            query = f'{song_title} by {artist_name} official audio'
            cmd = ["yt-dlp", f"ytsearch:{query}", "--print", "webpage_url"]

            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    url = result.stdout.strip().split("\n")[0]
                    return song_id, url
                else:
                    print(f"Error fetching URL for {song_id}: {result.stderr.strip()}")
            except Exception as e:
                print(f"Exception fetching URL for {song_id}: {e}")

        return song_id, ''
    
    # Use ThreadPoolExecutor to parallelize fetching URLs
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_url, batch)

    return dict(results)

def download_audio(song_id, url, folder_path):
    """Downloads the song from the URL to the specified folder path."""
    if not url:
        print(f"Skipping {song_id} due to missing URL")
        return

    os.makedirs(folder_path, exist_ok=True)  # Ensure directory exists
    output_template = os.path.join(folder_path, f'{song_id}.%(ext)s')  # Output format

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Downloaded: {url}")
        return os.path.join(folder_path, f'{song_id}.wav')
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def fetch_all_urls(batches, result):
    """Fetch URLs for all batches sequentially."""
    
    all_urls = {}
    for i, batch in enumerate(batches):
        start_time = time.time()
        
        batch_urls = fetch_urls_for_batch(batch, result)
        
        end_time = time.time()
        print(f'Batch {i + 1} completed in {end_time - start_time:.2f} seconds')

        all_urls.update(batch_urls)
    
    return all_urls    

def get_previews(result: dict, folder_path=None, download=True, batch_size=20, max_workers=4):
    """Process songs in batches to get preview URLs."""
    
    print('-----------------Getting Preview URLs-----------------')

    song_ids = result['songs']['song_id'].to_list()
    
    # Split song IDs into batches
    batches = [song_ids[i:i + batch_size] for i in range(0, len(song_ids), batch_size)]
    # print(batches)

    print(f'Processing {len(batches)} batch{"es" if len(batches) > 1 else ""}...')
    
    urls = fetch_all_urls(batches, result)

    # Assign fetched URLs to the dataset
    for song_id, url in urls.items():
        result['songs'].loc[result['songs']['song_id'] == song_id, 'preview_url'] = url

    if download:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(download_audio, song_id, url, folder_path) for song_id, url in urls.items()]
            for future in futures:
                future.result()

    return result

def top_tracks_for_recs(headers: dict, user_id, result: dict, num=10):
    result = get_top(headers=headers, user_id=user_id, result=result, limit=num, top_artists=False)
    
    folder_path = os.path.join(CURR_DIR, 'temp_downloads')
    get_previews(result, folder_path, batch_size=num)

    file_paths = [os.path.join(folder_path, file_path) for file_path in os.listdir(folder_path)]
    features = analyze_songs(file_paths)

    print(features)

    for file_path in file_paths:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except PermissionError:
            print(f"Permission denied: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    return result

def load_tables(tables=['users', 'songs', 'artists', 'artist_genres', 'user_song_interactions', 'user_artist_interactions', 'song_artist_interactions']):
    print('\nLoading Data...\n')
    connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(connection_string)

    result = {}

    for table in tables:
        query = f'SELECT * FROM {table}'
        df = pd.read_sql(query, engine)
        result[table] = df
    print('\nData successfully loaded\n')
    return result

def load_user_tables(user_id: str):
    query = '''
        SELECT 
            s.*,
            a.*
        FROM
            user_song_interactions usi
        JOIN 
            songs s ON usi.song_id = s.song_id
        JOIN 
            song_artist_interactions sai ON s.song_id = sai.song_id
        JOIN 
            artists a ON sai.artist_id = a.artist_id
        WHERE 
            usi.user_id = %s;
    '''

    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute(query, (user_id,))
    result = cursor.fetchall()
    cursor.close()
    conn.close()

def validate_audio_file(file_path):
    info = mediainfo(file_path)
    return 'duration' in info and float(info['duration']) > 0

def analyze_song(song_path: str):
    try:
        with audioread.audio_open(song_path) as audio:
            sr = audio.samplerate
            samples = np.frombuffer(b"".join(audio), dtype=np.int16).astype(np.float32)

        # Normalize audio
        samples /= np.max(np.abs(samples))

        # MFCC (13 coefficients) – Captures the audio's timbral texture.
        mfcc = lr.feature.mfcc(y=samples, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Chroma (12 bins) – Captures harmonic content and tonality.
        chroma = lr.feature.chroma_stft(y=samples, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Spectral Centroid – Helps distinguish brightness of sounds.
        spectral_centroid = lr.feature.spectral_centroid(y=samples, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)

        # Spectral Bandwidth – Provides insight into the texture of the sound.
        spectral_bandwidth = lr.feature.spectral_bandwidth(y=samples, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)

        # RMS Energy – Represents the loudness of the song, helping with dynamics.
        rms = lr.feature.rms(y=samples)
        rms_mean = np.mean(rms)

        # Tempo (BPM) – Key for understanding the song's rhythm and pace.
        onset_env = lr.onset.onset_strength(y=samples, sr=sr)
        tempo, _ = lr.beat.beat_track(onset_envelope=onset_env, sr=sr)
        
        # Zero-Crossing Rate – Useful for distinguishing between percussive and tonal elements.
        zero_crossing_rate = lr.feature.zero_crossing_rate(y=samples)
        zero_crossing_rate_mean = np.mean(zero_crossing_rate)

        song_id = os.path.splitext(os.path.basename(song_path))[0]

        features = np.concatenate([
            [song_id],
            mfcc_mean,
            chroma_mean,
            [spectral_centroid_mean],
            [spectral_bandwidth_mean],
            [rms_mean],
            tempo,
            [zero_crossing_rate_mean]
        ])

        return features

    except Exception as e:
        print(f"Unexpected error while analyzing {song_path}: {e}")
        return None

def analyze_songs(songs: pd.DataFrame, max_workers=None, prompt=False, batch_size=20):
    print('-----------------Analyzing Songs-----------------')

    columns = [
        'song_id', 'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 
        'mfcc_6', 'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10', 'mfcc_11', 
        'mfcc_12', 'mfcc_13', 
        'chroma_1', 'chroma_2', 'chroma_3', 'chroma_4', 'chroma_5', 'chroma_6', 
        'chroma_7', 'chroma_8', 'chroma_9', 'chroma_10', 'chroma_11', 'chroma_12',
        'spectral_centroid', 
        'spectral_bandwidth', 
        'rms', 
        'tempo',
        'zero_crossing_rate'
    ]
    

    batches = [songs['song_id'].to_list()[i:i+batch_size] for i in range(0, len(songs['song_id'].to_list()), batch_size)]

    print(f'{len(batches)} Batches')

    features = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i, batch in enumerate(batches):
            print(f'\nProcessing batch... {i + 1}')
            preview_urls = []
            song_ids = []
            for song_id in batch:
                try:
                    url = songs.loc[songs['song_id'] == song_id, 'preview_url'].dropna().values[0]
                    preview_urls.append(url)
                    song_ids.append(song_id)
                except IndexError:
                    print(f"Skipping {song_id} due to missing preview_url")

            if not preview_urls:
                continue
            
            downloads = list(executor.map(download_audio, song_ids, preview_urls, [os.path.join(CURR_DIR, 'temp_downloads')] * len(song_ids)))
            downloads = [download for download in downloads if download is not None and validate_audio_file(download)]
            batch_results = list(executor.map(analyze_song, downloads))

            for path in downloads:
                os.remove(path)

            # print(f'Batch len: {len(batch_results)}\nBatch Results: {batch_results}')
                
            count = 0
            for song_features in batch_results:
                # print(f"song_id: {song_features[0]}, Features: {song_features[1:]}")
                if song_features is not None:
                    count += 1
                    features.append(song_features)
            print(f'Features extracted for {count} song(s)')

    # print(f"Features list: {features}")
    song_features_df = pd.DataFrame(data=features, columns=columns)

    if prompt:
        while True:
            save = input('Save features to database? [y/n]')
            if save == 'y':
                add_df_to_db({'song_features':song_features_df})
                break
            if save == 'n':
                break
            else:
                print('Invalid answer')
    else:
        add_df_to_db({'song_features': song_features_df})

def sec_to_min(seconds):
    if seconds < 60:
        return f'{seconds} seconds'
    elif seconds < 3600:
        return f'{int(seconds/60)} minute{'s' if seconds/60 > 1 else ''} {seconds%60} seconds'
    else:
        ret = f'{int(seconds/60/60)} hour{'s' if seconds/60 > 1 else ''} '
        seconds /= 60
        ret += f'{int(seconds/60)} minute{'s' if seconds/60 > 1 else ''} {seconds%60} seconds'
        return ret
    


if __name__=='__main__':
    TOP_REC = False
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
    LOAD_DATA = False
    ARTIST_NAMES = False
    DOWNLOAD_SONGS = False
    SONG_FEATURES = False
    FEATURIZE_TOP_SONGS = False
    FEATURIZE_DATA_SET = False
    FEATURIZE_MISSING_SONGS = False
    LOAD_FEATURES = True
    

    test_token = os.getenv('USER_TEST_TOKEN')
    

    if TOP_REC:

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

        start_time = time.time()
        result = top_tracks_for_recs(headers, result['users']['user_id'].iloc[0], result)

        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')

        # print(result['songs']['preview_url'])

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

        result = get_top(headers, user_id, result)
        for key in result:
            result[key] = result[key].drop_duplicates()
        start_time = time.time()
        result = get_genres(headers, user_id, result)
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')

        print(f'Artist Genres:\n{len(result['artist_genres'])} rows\n{result['artist_genres']['artist_id'].nunique()} unique artist ID\'s\n{result['artist_genres'].head(3)}')

    if TABLES:
        start_time = time.time()
        result = get_init_tables(test_token)
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
        link = ''
        # search_youtube('Take Me Out', 'Franz Ferdinand')
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
        result = get_top(headers, user_id, result)
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
        result = get_all_saved_tracks(headers, user_id, result)
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
        result = get_all_playlist_tracks(headers, user_id, result, print_results=True)
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
        result = get_followed_artists(headers, user_id, result)
        end_time = time.time()
        print(f'Completed in {end_time - start_time} seconds')
        
        for key in result:
            df = result[key]
            print(f'{key}: {len(df)} rows\n{df.head(3)}\n')
    if SAVE_DATA:

        start_time = time.time()

        print(f'Token: {test_token}')

        result = get_init_tables(test_token)

        end_time = time.time()
        print(f'Total time: {end_time - start_time} seconds')

        for key in result:
            df = result[key]
            print(f'\n{key}:')
            print(f'{len(df)} entries')
            print(df.head(3), '\n')

        while True:
            save = input('Save to database? [y/n]')
            if save == 'y':
                add_df_to_db(result)
                break
            elif save == 'n':
                break
            else:
                print('Invalid input')

    if SAVE_PREVIEWS:

        result = load_tables(['songs', 'artists', 'song_artist_interactions'])
        result['songs'] = result['songs'].iloc[:200]
        # print(len(result['songs']))
        
        start_time = time.time()
        result = get_previews(result)
        end_time = time.time()
        print(f'\nCompleted in {end_time - start_time} seconds\n')
        
        print(result['songs']['preview_url'])

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
    
    if SONG_FEATURES:
        folder_path = os.path.join(CURR_DIR, 'temp_downloads')

        file_paths = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)]

        start_time = time.time()

        features = analyze_songs(file_paths)

        end_time = time.time()
        print(f'{len(features)} songs analyzed in {end_time - start_time} seconds')
    if FEATURIZE_TOP_SONGS:
        start_time = time.time()
        result = load_tables()

        result['songs'] = result['songs'].iloc[:10].copy()

        analyze_songs(result['songs'], prompt=True)

        end_time = time.time()
        print(f'\nElapsed time: {end_time - start_time} seconds\n')

    if FEATURIZE_DATA_SET:
        start_time = time.time()
        result = load_tables()

        analyze_songs(result['songs'])

        end_time = time.time()
        print(f'\nElapsed time: {end_time - start_time} seconds\n')

    if FEATURIZE_MISSING_SONGS:
        start_time = time.time()

        # query = '''
        #     WITH missing_songs AS (
        #         SELECT s.* FROM songs s
        #         LEFT JOIN song_features sf ON s.song_id = sf.song_id
        #         WHERE sf.song_id IS NULL
        #     )
        #     SELECT 
        #         ms.*,
        #         sai.artist_id,
        #         a.name
        #     FROM missing_songs ms
        #     LEFT JOIN song_artist_interactions sai ON ms.song_id = sai.song_id
        #     LEFT JOIN artists a ON sai.artist_id = a.artist_id;
        # '''

        # df = pd.read_sql(query, engine)

        # tables = {}
        # tables['songs'] = df.drop(columns=['artist_id', 'name'])
        # tables['song_artist_interactions'] = df[['song_id', 'artist_id']]
        # tables['artists'] = df[['artist_id', 'name']]

        query = '''
            SELECT s.* FROM songs s
            LEFT JOIN song_features sf ON s.song_id = sf.song_id
            WHERE sf.song_id IS NULL
        '''

        engine = get_engine()

        songs = pd.read_sql(query, engine)

        print(f'Rows: {len(songs)}\n{songs['title']}')
        analyze_songs(songs)

        end_time = time.time()
        print(f'Completed in {sec_to_min(end_time-start_time)}')

    if LOAD_FEATURES:
        query = '''
            SELECT * FROM song_features
        '''
        connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
        engine = create_engine(connection_string)
        song_features = pd.read_sql(query, engine)
        print(song_features)