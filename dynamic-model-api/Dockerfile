FROM python:3.9-slim

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
# force CPU torch installation
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app
COPY . .

ENV PYTHONUNBUFFERED=1
# Expose the port the app runs on
EXPOSE 5000

# Run your app with uvicorn
CMD ["uvicorn", "app:asgi_app", "--host", "0.0.0.0", "--port", "5000"]

# this setup is for CPU usage (not GPU)

# ----- DOCKER INSTRUCTIONS: -----
# OPEN docker destop
# NAVIGATE TO scraply/dynamic-model-api
# COMMAND TO RUN in terminal:
# docker build -t scraply-api:lean .
# docker run --rm -p 5000:5000 scraply-api:lean

# COMMAND TO RUN in terminal: (if you do not want to use docker)
# uvicorn app:socket_app --host 0.0.0.0 --port 5000 --reload