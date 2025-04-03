from utils import analyze_missing_songs, RateLimitException
from datetime import datetime, timedelta, timezone
from celery_config import celery
import logging


@celery.task(
    name='song_tasks.analyze_user_songs', 
    queue='song_queue',
    bind=True
)
def analyze_user_songs(self, user_id: str, retries: int = 0):
    try:
        analyze_missing_songs(user_id)
        return {'message': f'Analysis of user {user_id} library complete'}
    except RateLimitException:
        # Start at 10 minutes and then increase exponentially
        countdown = 600 * (2 ** retries)  # Exponential backoff starting at 10 minutes
        if countdown > 7200:  # Limit the maximum countdown to 2 hour
            countdown = 7200

        eta_time = datetime.now(timezone.utc) + timedelta(seconds=countdown)

        logging.info(f'Rate limit detected for user {user_id}. Trying again at {eta_time}.')
        self.apply_async(args=[user_id, retries + 1], eta=eta_time)
        