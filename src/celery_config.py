from dotenv import load_dotenv
from celery import Celery
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# load_dotenv()
REDIS_URL = 'redis://localhost:6379/0'

celery = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["song_tasks", "user_tasks"]
)
