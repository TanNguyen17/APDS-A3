# Use an official Python runtime as a parent image
FROM python:3.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data required for preprocessing
RUN python -m nltk.downloader wordnet omw-1.4

# Copy the rest of the application code
# Note: large files are excluded via .dockerignore
COPY . .

# Ensure the models directory exists (it will be the mount point)
RUN mkdir -p /app/models

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]
