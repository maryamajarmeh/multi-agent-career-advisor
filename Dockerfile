# syntax=docker/dockerfile:1

FROM python:3.11-slim

WORKDIR /app

# System deps some Python packages (e.g. pypdf, langchain extras) may need
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer is cached unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the project
COPY . .

# FastAPI (8000) and Streamlit (8501) both live in this image;
# docker-compose picks which one to run per-container via `command:`
EXPOSE 8000 8501

# Default command if this image is run standalone (backend)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
