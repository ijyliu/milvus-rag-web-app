# Image for the Flask application

# Need Ubuntu for pymilvus
#FROM ubuntu:24.10

# Install pip
#RUN apt-get update && apt-get install -y python3-pip

# pymilvus specialized image
FROM bitnami/pymilvus:2.4.3

# Exit python
RUN exit

# Set working directory to root
WORKDIR /

# Install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --break-system-packages --requirement /tmp/requirements.txt

# Copy the API credentials
COPY ./Credentials /Credentials

# Copy the application code and files
COPY ./App /App

# Set the working directory
WORKDIR /App

# Run the application on port 8080
EXPOSE 8080
ENTRYPOINT [ "python", "app.py" ]
