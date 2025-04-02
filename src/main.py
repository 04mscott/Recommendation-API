from fastapi import FastAPI, HTTPException, Header
# import celery_worker
from celery.result import AsyncResult
from pydantic import BaseModel
from dotenv import load_dotenv
from celery import Celery
import recommend, utils
import traceback
import logging
import os


app = FastAPI()

celery = Celery("worker", broker="redis://localhost:6379/0", backend='redis://localhost:6379/0')

class User(BaseModel):
    user_id: str

class Song(BaseModel):
    song_id: str
    title: str
    artists: str
    img_url: str
    preview_url: str

class Recommendations(BaseModel):
    user_id: str
    total: int
    songs: list[Song]

def validate_fastapi_token(token: str):
    load_dotenv()
    valid_token = os.getenv('SECRET_TOKEN')
    if token != valid_token:
        raise HTTPException(status_code=403, detail="Invalid API access token")

@celery.task
def save_user_data(
    user_id: str,
    fastapi_token: str = Header(..., alias="Authorization"),  # Required FastAPI token
    spotify_token: str | None = Header(None, alias="Spotify-Token")
):

    print(f"Started task for user {user_id}")
    validate_fastapi_token(fastapi_token)
    
    user_exists = recommend.check_user_time(user_id, time=False)

    try:
        
        if user_exists:
            up_to_date = recommend.check_user_time(user_id)
            if up_to_date:
                print(f"User {user_id} up to date")
                return {'message': f'User {user_id} up to date'}
            else:
                # recommend.update_user(user_id, spotify_token)
                print(f"Successfully updated user data for {user_id}")
                return {'message': f'User {user_id} updated'}
            
        else:
            tables = utils.get_init_tables(spotify_token)
            utils.add_df_to_db(tables)
            utils.analyze_missing_songs(user_id)
            print(f"Successfully saved user data for {user_id}")
            return {'message': f'Successfully saved user {user_id}'}
    
    except HTTPException as http_err:
        logging.warning(f"HTTP error for user {user_id}: {http_err.detail}")
        raise

    except Exception as e:
        logging.error(f"Error saving data for user {user_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail='An unexpected error occurred while saving user data.')




@app.get("/status/{task_id}")
def get_task_status(task_id: str):
    task = AsyncResult(task_id, app=celery)
    if task.state == 'PENDING':
        return {"status": "Task is still processing"}
    elif task.state == 'SUCCESS':
        return {"status": "Task completed", "result": task.result}
    else:
        return {"status": task.state}

@app.get('/recommend', response_model = Recommendations)
def get_recommendation(
    user: User, 
    fastapi_token: str = Header(..., alias="Authorization"),  # Required FastAPI token
    spotify_token: str | None = Header(None, alias="Spotify-Token")
) -> Recommendations:
    
    logging.info("User request received")
    validate_fastapi_token(fastapi_token)

    user_id = user.user_id
    generic = False
    recs = []
    
    try:
        if not recommend.check_user_time(user_id):
            try:
                result = utils.top_tracks_for_recs(spotify_token, user_id)
                if result != 1:
                    generic = True
                    recs = recommend.recommend_songs()
            except Exception as e:
                print(f'Error getting top tracks for user {user_id} (user likely has no top tracks): {e}')
                
        if not generic:
            recs = recommend.recommend_songs(user_id)

        logging.info(f"Successfully retrieved {'generic' if generic else 'personalized'} recommendations for user {user_id}")
        return {
                'user_id': user_id,
                'total': len(recs),
                'songs': recs
            }
    
    except HTTPException as http_err:
        logging.warning(f"HTTP error for user {user_id}: {http_err.detail}")
        raise

    except Exception as e:
        logging.error(f"Error recommending songs for user {user_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail='An unexpected error occurred while recommending songs.')

@app.post('/save-data')
def save_data(
    user: User,
    fastapi_token: str = Header(..., alias="Authorization"),  # Required FastAPI token
    spotify_token: str | None = Header(None, alias="Spotify-Token")
):
    user_id = user.user_id
    task = save_user_data.apply_async((user_id, fastapi_token, spotify_token))
    return {"message": "Task started", "task_id": task.id}
    # return {'message': f'Saving data for user {user_id}'}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


