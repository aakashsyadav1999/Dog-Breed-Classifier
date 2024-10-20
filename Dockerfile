# Use the official Python 3.10 image as the base image
FROM python:3.10-slim-buster


# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq-dev \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy dependencies and install them
COPY requirements.txt . 
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY . .

# Expose the port and set the entry point
EXPOSE 5000
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]