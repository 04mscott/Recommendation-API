# 🎵 Music Recommender API

A high-performance, asynchronous music recommendation system that integrates Spotify and YouTube APIs to deliver personalized track suggestions using semantic similarity. Built with FastAPI, Celery, Redis, and Docker, this API is designed to work seamlessly with a custom **React + Spring Boot frontend application** as part of a full-stack music recommendation platform.

---

## 🚀 Overview

This project serves as the backend engine for a full-stack music recommender system. It takes in a Spotify track and returns similar songs by leveraging audio embeddings, cosine similarity, and external API lookups. The system is built for scalability, fast response time, and integration with modern client-side applications.

---

## 🧠 Core Features

- ⚡ **FastAPI Backend:** Lightweight and async-ready API endpoints
- 🧵 **Celery Task Queue:** Handles time-consuming Spotify/YouTube processing in the background
- 🧠 **Embedding-Based Similarity:** Finds semantically similar songs using vector math
- 🔁 **YouTube Integration:** Retrieves video links for recommended tracks with rate-limiting support
- 🧊 **Redis:** Used for task management, exponential backoff, and API key retries
- 🐬 **MySQL Storage:** Persists user input and recommendation metadata
- 📦 **Dockerized:** Production-ready deployment setup with containerization
- 🔒 **Token Authentication:** Protects endpoints from unauthorized access

---

## 📂 Project Structure
```bash
Recommendation-API
├── src/
│   ├── utils.py           # Interacting with Spotify API and saving data to database
│   ├── recommend.py       # Embedding, similarity, and helper logic
│   ├── celery_config.py   # Celery environment setup
│   ├── main.py            # FastAPI entrypoint
│   └── celery_tasks.py    # Celery tasks for async computation
├── .dockerignore          # Ignore files
├── .gitignore             # 
├── compose.yml            # Configuration specs for Docker
├── Dockerfile             # Setup for Docker
├── README.md              # What you're looking at right now!
└── requirements.txt       # All required libraries/packages
```
---

## 🧪 Example Use Case

1. A new user logs into the BeatMatch app.
2. The Spring Boot backend sends a `POST` request to the `/save-data` endpoint.
4. A Celery task is triggered to:
   - Fetch all songs associated with user's Spotify acount
   - Search YouTube API for official song audios for preview playback and analysis
   - Store the full result set in MySQL
   - Trigers a new Celery task to analyze the songs saved for the given user
   - Songs are process in batches
   - Temporarily downloaded, analyzed using librosa, stored in MySQL, remove downloaded files
5. Once ready, the frontend hits `/recommend` to retrieve the final list of tracks + YouTube links.
   - In case user data is not available/taking too long, general recommendations are given using the mean feature vector of the entire dataset
6. The backend polls `/status/{task_id}` to check when processing is complete and notifies user when their recommendations are ready to be fully personalized.

---

## 📈 Future Improvements

- Add JWT-based multi-user authentication
- Introduce collaborative filtering or hybrid models
- Expand to support genres or mood-based filtering
- Integrate with playlist-building and export features

---

## 🧑‍💻 Author

**Mason Scott**  
Third year CS major (Data Science track), Statistics minor  
University of Maryland – College Park  
🌐 Website: [masonscott.net](https://masonscott.net)  
🐙 GitHub: [@04mscott](https://github.com/04mscott)  
🔗 LinkedIn: [Mason T. Scott](https://www.linkedin.com/in/mason-t-scott/)

---

## 📌 Notes

> This API was built specifically to serve a React + Spring Boot app for a full-stack music recommendation system. It is designed to run in Docker containers with background task support and optimized external API interaction. While setup and deployment details are not included in this README, the system is fully containerized and production-ready.
