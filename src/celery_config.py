from dotenv import load_dotenv
from celery import Celery
import sys
import os

# Add the src directory to the Python path
sys.path.append('/app/src')

load_dotenv()
REDIS_URL = os.getenv('REDIS_URL', default='redis://localhost:6379/0')

celery = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["celery_tasks"]
)
