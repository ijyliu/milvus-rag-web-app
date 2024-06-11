# Image for the Flask application

# Use an official Python runtime as a parent image
FROM adrianliechti/sentence-transformers:bge-base-en-v1

# Install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt

# Copy the application code and files
COPY . /app
WORKDIR /app

# Run the application on port 8080
EXPOSE 8080
CMD ["python", "app.py"]
