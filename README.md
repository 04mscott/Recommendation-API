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
src/ ├── api/ # FastAPI route handlers │ ├── recommend.py │ └── save_data.py ├── celery_worker/ # Celery background tasks │ └── analyze_and_store.py ├── utils/ # Embedding, similarity, and helper logic ├── config.py # Environment setup ├── main.py # FastAPI entrypoint └── celery_worker.py # Celery entrypoint
```
---

## 🧪 Example Use Case

1. A user selects a Spotify track in the React frontend.
2. The frontend sends a `POST` request to the `/save-data` endpoint.
3. A Celery task is triggered to:
   - Fetch audio features from Spotify
   - Generate a list of similar songs
   - Search for related YouTube videos
   - Store the full result set in MySQL
4. The frontend polls `/status/{task_id}` to check when processing is complete.
5. Once ready, the frontend hits `/recommend` to retrieve the final list of tracks + YouTube links.

---

## 📈 Future Improvements

- Add JWT-based multi-user authentication
- Introduce collaborative filtering or hybrid models
- Implement caching for commonly requested songs
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

> This API was built specifically to serve a React + Spring Boot frontend app for a full-stack music recommendation system. It is designed to run in Docker containers with background task support and optimized external API interaction. While setup and deployment details are not included in this README, the system is fully containerized and production-ready.
