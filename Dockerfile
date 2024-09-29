FROM python:3.11

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get install ffmpeg

COPY ./app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
