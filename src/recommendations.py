from sqlalchemy import create_engine, text
from collections import defaultdict
from dotenv import load_dotenv
from requests import post, get
import mysql.connector
import pandas as pd
import numpy as np
import base64
import json
import time
import os

load_dotenv()

client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_name = os.getenv("DB_NAME")
youtube_key = os.getenv('YOUTUBE_KEY')