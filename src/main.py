from fastapi import FastAPI, HTTPException, Header
from celery_tasks import save_user_data
from celery.result import AsyncResult
from celery_config import celery
from pydantic import BaseModel
import recommend, utils
import traceback
import logging

app = FastAPI()

''' MODELS '''
class User(BaseModel):
    user_id: str

class Recommendation(BaseModel):
    user_id: str
    total: int
    song_ids: list[str]

class Message(BaseModel):
    message: str
    task_id: str = 'None'
    
class Status(BaseModel):
    message: str

class Stats(BaseModel):
    songs_analyzed: int = 0
    num_recs: int = 0
    recent_like: str | None = None
    num_likes: int = 0
    percent: int = 0

''' API ENDPOINTS'''
@app.get('/ping')
def ping():
    return {'message', 'pong from rec engine api'}

@app.get("/status/{task_id}", response_model = Status)
def get_task_status(task_id: str):
    task = AsyncResult(task_id, app=celery)
    if task.state == 'PENDING':
        return {"status": "Task is still processing"}
    elif task.state == 'SUCCESS':
        return {"status": "Task completed", "result": task.result}
    else:
        return {"status": task.state}

@app.get('/recommend', response_model = Recommendation)
def get_recommendation(
    user: User, 
    fastapi_token: str = Header(..., alias="Authorization"),
    spotify_token: str | None = Header(None, alias="Spotify-Token")
) -> Recommendation:
    
    logging.info("User request received")
    utils.validate_fastapi_token(fastapi_token)

    user_id = user.user_id
    generic = False
    recs = {}
    
    try:
        if not recommend.check_user_time(user_id):
            try:
                result = utils.top_tracks_for_recs(spotify_token, user_id)
                if result != 1:
                    generic = True
                    recs = recommend.recommend_songs()
            except Exception as e:
                logging.info(f'Error getting top tracks for user {user_id} (user likely has no top tracks): {e}')
                
        if not generic:
            recs = recommend.recommend_songs(user_id)

        logging.info(f"Successfully retrieved {'generic' if generic else 'personalized'} recommendations for user {user_id}")
        return {
                'user_id': user_id,
                'total': len(recs['song_ids']),
                'song_ids': recs['song_ids']
            }
    
    except HTTPException as http_err:
        logging.warning(f"HTTP error for user {user_id}: {http_err.detail}")
        raise

    except Exception as e:
        logging.error(f"Error recommending songs for user {user_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail='An unexpected error occurred while recommending songs.')

@app.get('/get-stats/{user_id}', response_model = Stats)
def get_stats(
    user_id: str, 
    fastapi_token: str = Header(..., alias="Authorization")
) -> Stats:
    
    logging.info("User request received")
    utils.validate_fastapi_token(fastapi_token)
    return recommend.get_user_stats(user_id)

@app.post('/save-data', response_model = Message)
def save_data(
    user: User,
    fastapi_token: str = Header(..., alias="Authorization"),
    spotify_token: str | None = Header(None, alias="Spotify-Token")
) -> Message: 
    user_id = user.user_id
    task = save_user_data.apply_async(args=[user_id, fastapi_token, spotify_token])
    return {"message": "Task started", "task_id": task.id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)

