# syntax=docker/dockerfile:1
# Use Amazon ECR Public mirror to avoid docker.io mirror issues
FROM public.ecr.aws/docker/library/python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_NO_CACHE_DIR=on

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better layer cache)
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8000

# Default envs (can override via docker run -e ...)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    DB_URL=sqlite:////data/app.db \
    APP_SECRET=change_me

# Runtime user and volume for data
RUN useradd -m appuser
RUN mkdir -p /data && chown -R appuser:appuser /data
VOLUME ["/data"]

USER appuser

# Run
CMD ["python", "main.py"]
