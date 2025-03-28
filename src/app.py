from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
import asyncio
import threading

app = FastAPI()

# Model for incoming data (user's access token)
class UserData(BaseModel):
    user_token: str

# Example model for recommendations
class RecommendationsResponse(BaseModel):
    recommendations: list
    user_token: str

# Endpoint to get recommendations
@app.post("/recommendations", response_model=RecommendationsResponse)
async def get_recommendations(user_data: UserData):
    access_token = user_data.user_token

    # 1. Check if the top 10 tracks are already in the database
    top_tracks = await fetch_top_tracks(access_token)

    # If the tracks need to be fetched, downloaded, and vectorized:
    if not top_tracks:  # This means no data found in the database
        top_tracks = await fetch_and_process_top_tracks(access_token)

    # 2. Generate recommendations based on the top tracks
    recommendations = await generate_recommendations(top_tracks, access_token)

    return {"recommendations": recommendations, "user_token": access_token}

# Function to simulate checking the database for the top tracks
async def fetch_top_tracks(access_token: str):
    # Simulate checking the database for the top tracks of the user.
    # This would query your MySQL database to see if tracks exist.
    # In reality, use a database query here.
    return []  # Empty list indicates no tracks found in the database

# Function to fetch, download, and vectorize top tracks
async def fetch_and_process_top_tracks(access_token: str):
    # Fetch the user's top tracks from Spotify
    top_tracks = await get_user_top_tracks(access_token)
    
    # Download, process, and vectorize the tracks
    await download_and_vectorize_tracks(top_tracks)
    
    # Save the tracks to the database (not blocking the API)
    await save_tracks_to_db(top_tracks)
    
    return top_tracks

# Function to fetch top tracks from Spotify
async def get_user_top_tracks(access_token: str):
    url = "https://api.spotify.com/v1/me/top/tracks?limit=10"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail="Failed to fetch top tracks from Spotify")

    return [track["id"] for track in response.json().get("items", [])]

# Function to download and vectorize the tracks asynchronously
async def download_and_vectorize_tracks(top_tracks):
    # You can implement actual download and vectorization logic here
    # For example, using yt-dlp to download audio files, and then vectorizing them
    for track_id in top_tracks:
        # Simulate the downloading and vectorization process
        await asyncio.sleep(1)  # Replace with real downloading logic

# Function to save tracks to the database asynchronously
async def save_tracks_to_db(top_tracks):
    # Simulate saving tracks to the database
    # In reality, perform a database insert here
    await asyncio.sleep(1)  # Simulating asynchronous DB save

# Function to generate recommendations based on top tracks
async def generate_recommendations(top_tracks, access_token):
    
    return None

# Endpoint to save data asynchronously
@app.post("/save_data")
async def save_data(user_data: UserData, background_tasks: BackgroundTasks):
    # Save data to database asynchronously
    background_tasks.add_task(save_user_data, user_data.user_token)
    return {"message": "Data saving started", "status": "processing"}

# Simulate saving user data to the database
async def save_user_data(user_token: str):
    # Here, you would save the user’s top tracks, artists, etc., to the database
    await asyncio.sleep(3)  # Simulating a long DB operation
    print(f"User data for {user_token} saved to the database.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
