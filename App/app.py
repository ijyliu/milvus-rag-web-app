# app.py
# Milvus RAG web app
# Put a 'Credentials' folder in the root of this repository with the following files:
# Note: Place your Zilliz URI in a file called 'zilliz_uri.txt'.
# Note: Place your Zilliz token in a file called 'zilliz_token.txt'.
# Note: Place your Google API key in a file called 'google_api_key.txt'.
# Note: PLace your Mixedbread API key in a file called 'mixedbread_api_key.txt'.

##################################################################################################

# Packages
from pymilvus import Collection, connections
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from RAG_Functions import *
import google.generativeai as genai
from mixedbread_ai.client import MixedbreadAI

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

##################################################################################################

# Running the app

##################################################################################################

# Setting up the flask app
app = Flask(__name__)
CORS(app)

# Load Google API key
with open(os.path.expanduser('../Credentials/google_api_key.txt')) as f:
    GOOGLE_API_KEY = f.read().strip()
genai.configure(api_key=GOOGLE_API_KEY)

# Load mixedbread API key
with open(os.path.expanduser('../Credentials/mixedbread_api_key.txt')) as f:
    MIXEDBREAD_API_KEY = f.read().strip()

# Setup MixedbreadAI client
mxbai_client = MixedbreadAI(api_key=MIXEDBREAD_API_KEY)

# Setup Chat Model
chat_model = genai.GenerativeModel('gemini-1.0-pro')

# Decorator to get function called when POST request sent to /chat
@app.route('/chat', methods=['POST'])
def chat():
    # Load input text from json posted
    user_input = request.json.get('chat')
    # Error if input is empty
    if not user_input:
        return jsonify({"error": "Empty input text"}), 400
    # Get message from Gemini Pro Chat
    message = gemini_pro_chat_response(user_input, mxbai_client, chat_model, collection)
    # Return message as json
    return jsonify({"response": True, "message": message})

# Render loading page
@app.route('/')
def home():
    return render_template('loading.html')

# Redirect to index
@app.route('/main')
def index():
    return render_template('index.html')

# Serve the app with gevent
if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
