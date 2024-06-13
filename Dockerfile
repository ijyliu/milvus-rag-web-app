# Image for the Flask application

# Need Ubuntu for pymilvus
FROM ubuntu:24.10

# Install pip
RUN apt-get update && apt-get install -y python3-pip

# Install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --break-system-packages --requirement /tmp/requirements.txt

# Copy the application code and files
COPY . /app
WORKDIR /app

# Run the application on port 8080
EXPOSE 8080
CMD ["python3", "app.py"]
