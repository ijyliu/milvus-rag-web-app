# app.py
# Milvus RAG web app
# Put a 'Credentials' folder in the root of this repository
# Place your Zilliz URI in a file called 'zilliz_uri.txt'
# Place your Zilliz token in a file called 'zilliz_token.txt'

##################################################################################################

# Packages
from pymilvus import Collection, connections
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from RAG_Functions import *

##################################################################################################

# Local flag
# Changes connection settings to a local Milvus instance
local = False

##################################################################################################

# Milvus setup

# Load URI from '.../Credentials/zilliz_uri.txt'
with open(os.path.expanduser('../Credentials/zilliz_uri.txt')) as f:
    zilliz_uri = f.read().strip()

# Load token from '.../Credentials/zilliz_token.txt'
with open(os.path.expanduser('../Credentials/zilliz_token.txt')) as f:
    zilliz_token = f.read().strip()

# Connect to Milvus
if local:
    connections.connect("default", host="localhost", port="19530")
if not local:
    connections.connect(alias="default", uri=zilliz_uri, token=zilliz_token)

# Set up collection name
collection_name = "text_embeddings"

print('set up collection name')

##################################################################################################

# Load the collection and create index if it does not exist

# Get collection
collection = Collection(name=collection_name)

# Check if index exists, create if it does not
if not collection.has_index():
    # Create Euclidean L2 index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

# Load the collection
collection.load()

print('loaded collection')

##################################################################################################

# Running the app

##################################################################################################

# Setting up the flask app
app = Flask(__name__)
CORS(app)

# Decorator to get function called when POST request sent to /chat
@app.route('/chat', methods=['POST'])
def chat():
    print('accepting user input')
    # Load input text from json posted
    user_input = request.json.get('chat')
    # Error if input is empty
    if not user_input:
        return jsonify({"error": "Empty input text"}), 400
    print('got user input')
    # Get message from gemma chat
    message = gemma_chat_response(user_input, collection)
    # Return message as json
    return jsonify({"response": True, "message": message})

# Function to get the environment variable with a default
def get_environment():
    """Gets the ENVIRONMENT variable, defaulting to 'local' if not set."""
    return os.environ.get('ENVIRONMENT', 'local')

# Render the index.html front end, passing the variable
@app.route('/')
def index():
    environment = get_environment()
    return render_template('index.html', environment=environment)

# Serve the app with gevent
if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
