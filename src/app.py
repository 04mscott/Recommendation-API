from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import recommend


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

@app.get('/')
def root():
    return {'Hello': 'World'}

# curl -X GET -H "Content-Type: application/json" '{"user_id":"..."}' 'http://127.0.0.1:8000/recommend'

@app.get('/recommend', response_model = Recommendations)
def get_recommendation(user: User) -> Recommendations:
    user_id = user.user_id
    if recommend.check_user(user_id):
        recs = recommend.recommend_songs(user_id)
        return {
            'user_id': user_id,
            'total': len(recs),
            'songs': recs
        }
    else:
        raise HTTPException(status_code=404, detail=f'user_id {user_id} not found')


@app.post('/save-data/{type}')
def save_data(type: str, token: str):
    if type == 'new_user':
        return {
            'new_user': 'saving data'
        }
    elif type == 'top_songs':
        return {
            'top_songs': 'saving top_songs'
        }
    else:
        raise HTTPException(status_code=400, detail=f"Invalid extension '{type}'")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
