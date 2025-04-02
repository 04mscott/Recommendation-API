from fastapi import FastAPI, HTTPException, Header
from main import validate_fastapi_token, User
from dotenv import load_dotenv
from celery import Celery
import recommend, utils
import traceback
import logging
import os

load_dotenv()

redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "worker",
    broker=redis_url,
    backend=redis_url
)

@celery_app.task
def save_user_data(
    user: User,
    fastapi_token: str = Header(..., alias="Authorization"),  # Required FastAPI token
    spotify_token: str | None = Header(None, alias="Spotify-Token")
):
    
    user_id = user.user_id

    import time
    time.sleep(10)  # Simulate long processing
    print(f"User {user_id} data processed.")
    return f"Finished processing user {user_id}"

    # logging.info("User request received")
    # validate_fastapi_token(fastapi_token)
    
    # user_exists = recommend.check_user_time(user_id, time=False)

    # try:
        
    #     if user_exists:
    #         up_to_date = recommend.check_user_time(user_id)
    #         if up_to_date:
    #             logging.info(f"User {user_id} up to date")
    #             return {'message': f'User {user_id} up to date'}
    #         else:
    #             recommend.update_user(user_id, spotify_token)
    #             logging.info(f"Successfully updated user data for {user_id}")
    #             return {'message': f'User {user_id} updated'}
            
    #     else:
    #         tables = utils.get_init_tables(spotify_token)
    #         utils.add_df_to_db(tables)
    #         utils.analyze_missing_songs(user_id)
    #         logging.info(f"Successfully saved user data for {user_id}")
    #         return {'message': f'Successfully saved user {user_id}'}
    
    # except HTTPException as http_err:
    #     logging.warning(f"HTTP error for user {user_id}: {http_err.detail}")
    #     raise

    # except Exception as e:
    #     logging.error(f"Error saving data for user {user_id}: {e}\n{traceback.format_exc()}")
    #     raise HTTPException(status_code=500, detail='An unexpected error occurred while saving user data.')