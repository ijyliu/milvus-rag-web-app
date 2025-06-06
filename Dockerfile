# Image for the Flask application

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

# Real-time logging
ENV PYTHONUNBUFFERED=1

# Run the application on port 8080
EXPOSE 8080
# Streamlit run command for app.py
# Use executable
ENTRYPOINT ["/.local/bin/streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0", "--server.enableCORS=true", "--server.enableXsrfProtection=true"]
# # keep container running
# ENTRYPOINT ["sh", "-c", "while true; do sleep 1; done"]
