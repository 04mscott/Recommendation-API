# Use official Python image
FROM python:3.11

# Set the working directory
WORKDIR /app

# Copy only requirements first (for caching efficiency)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire 'app' directory into the container
COPY app /app/

# Expose FastAPI's default port
EXPOSE 8000

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
