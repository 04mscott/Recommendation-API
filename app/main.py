from fastapi import FastAPI, HTTPException, Header
from celery_worker import save_user_data
from pydantic import BaseModel
from dotenv import load_dotenv
import recommend, utils
import traceback
import logging
import os


app = FastAPI()
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

''' Test Request:
curl -X GET "http://127.0.0.1:8000/recommend" \
     -H "Content-Type: application/json" \
     -H "Authorization: ly2DGr@ZOX5.3-?ve2Gq" \
     -H "Spotify-Token: BQATg8EHkoThVgP_wCZpwcpTMqq99ZDCIWPsWO4PIdwJJaSgQBgF7_00uDtYIBhdHzosBm-TcYSh4OURZzSE19T_xCqv4knz_KvX4NiBIUbKj2zU2s01mLVAf7WY3eRWBU3XaVwdxP7yJdG0u5vVPXGxY7ffwWxCIMW7FHpCS7FiXyoB1UlPvGgRwdL2vf3Rbk5dJZ1ZZu2obBWsptvcYpIsxtwoG7cdzO4v7PNofDAX_eBnvpQg4iG5Jloq-u1sG5oT6MKNgBvehqGioIm3Xk7y" \
     -d '{"user_id": "jzzkjx1n9mnb2hvk6r5q32n48"}'
'''

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
            result = utils.top_tracks_for_recs(spotify_token, user_id)
            if result != 1:
                generic = True
                recs = recommend.recommend_songs()
                
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
    task = save_user_data(user, fastapi_token, spotify_token).delay(user_id)
    return {"message": f"Processing user {user_id} in background", "task_id": task.id}
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
