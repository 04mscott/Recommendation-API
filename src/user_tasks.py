from .utils import get_init_tables, add_df_to_db, validate_fastapi_token
from fastapi import Header, HTTPException
from .recommend import check_user_time
from .celery_config import celery
import traceback
import logging


@celery.task(name='user_tasks.save_user_data', queue="user_queue")
def save_user_data(
    user_id: str,
    fastapi_token: str = Header(..., alias="Authorization"),
    spotify_token: str | None = Header(None, alias="Spotify-Token")
) -> dict[str, list[str]]:
    
    from .song_tasks import analyze_user_songs

    logging.info(f"Started task for user {user_id}")
    validate_fastapi_token(fastapi_token)
    
    user_exists = check_user_time(user_id, time=False)

    try:
        
        if user_exists:
            up_to_date = check_user_time(user_id)
            if up_to_date:
                logging.info(f"User {user_id} up to date")
                return {'message': f'User {user_id} up to date'}
            else:
                try:
                    tables = get_init_tables(user_id, spotify_token, update=True)
                    add_df_to_db(tables)
                    logging.info(f"Successfully updated user data for {user_id}")
                    task = analyze_user_songs.apply_async((user_id,))
                    return {
                        'messages': [
                            {'message': f'User {user_id} updated'}, 
                            {'message': f'Task started, task_id: {task.id}'}
                        ]
                    }
                except Exception as e:
                    logging.error(f"Error updating data for user {user_id}: {e}\n{traceback.format_exc()}")
                    raise HTTPException(status_code=500, detail='An unexpected error occurred while updating user data.')
            
        else:
            tables = get_init_tables(user_id, spotify_token)
            add_df_to_db(tables)
            logging.info(f"Successfully saved user data for {user_id}")
            task = analyze_user_songs.apply_async((user_id,))
            return {
                'messages': [
                    {'message': f'Successfully saved user {user_id}'}, 
                    {'message': f'Task started, task_id: {task.id}'}
                ]
            }
  
    
    except HTTPException as http_err:
        logging.warning(f"HTTP error for user {user_id}: {http_err.detail}")
        raise

    except Exception as e:
        logging.error(f"Error saving data for user {user_id}: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail='An unexpected error occurred while saving user data.')