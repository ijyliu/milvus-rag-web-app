# app.py
# Milvus RAG web app
# Note: Place your Zilliz URI in the Zilliz URI and Token folder of the directory this code is in, as the contents of a file called 'zilliz_uri.txt'.
# Note: Place your Zilliz token in the Zilliz URI and Token folder of the directory this code is in, as the contents of a file called 'zilliz_token.txt'.
# Note: Place your Google API key in the Google API Key folder of the directory this code is in, as the contents of a file called 'google_api_key.txt'.

##################################################################################################

# Packages
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import pandas as pd
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
from RAG_Functions import *
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os

##################################################################################################

# Local flag
# Changes connection settings to a local Milvus instance
local = False

##################################################################################################

# Milvus setup

# If collection does not already exist, load embeddings and create index
# Otherwise, load the collection

##################################################################################################

# Load URI from 'Zilliz URI and Token/zilliz_uri.txt'
with open(os.path.expanduser('./Zilliz URI and Token/zilliz_uri.txt')) as f:
    zilliz_uri = f.read().strip()

# Load token from 'Zilliz URI and Token/zilliz_token.txt'
with open(os.path.expanduser('./Zilliz URI and Token/zilliz_token.txt')) as f:
    zilliz_token = f.read().strip()

# Connect to Milvus
if local:
    connections.connect("default", host="localhost", port="19530")
if not local:
    connections.connect(alias="default", uri=zilliz_uri, token=zilliz_token)

# Set up collection name
collection_name = "text_embeddings"

# Prepare the collection if it does not already exist
if collection_name not in utility.list_collections():

    # fields: sentences, embeddings, companies, and documents
    fields = [
        FieldSchema(name="sentence_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # we got some long sentences in here so length for this field has to be quite long to accomodate some outliers
        FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=2**15),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=64),
    ]

    # Create schema
    schema = CollectionSchema(fields, description="Generated text embeddings")

    # Create collection
    collection = Collection(name=collection_name, schema=schema)

    # Location of the embeddings
    dir_path = './Embeddings'

    # Insert the embeddings into the collection
    for path in os.listdir(dir_path):

        # Clean names
        curr_company_name = path.split('_')[0]
        curr_document_name = path.split('_')[1].split('.')[0]

        # Read the parquet file
        df = pd.read_parquet(f'{dir_path}/{path}')

        # Get the sentences and embeddings
        sentences = df['sentence'].to_list()
        embeddings = df.filter(regex='^embed_element_').values.tolist()

        # Fill in the company and document name with one entry per sentence
        company_names = np.full(len(sentences), curr_company_name)
        document_names = np.full(len(sentences), curr_document_name)

        # Display any long names
        if len(curr_company_name) > 64:
            print(curr_company_name)
        if len(curr_document_name) > 64:
            print(curr_document_name)
        
        # Insert the data into the collection
        mr = collection.insert(
            [
                sentences,      # sentences
                embeddings,     # embeddings
                company_names,  # company names
                document_names  # document names
            ]
        )
    
    # Create Euclidean L2 index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    # Load the collection
    collection.load()

# If the collection already exists, load the collection
else:

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

# Load API key
with open(os.path.expanduser('./Google API Key/google_api_key.txt')) as f:
    GOOGLE_API_KEY = f.read().strip()
genai.configure(api_key=GOOGLE_API_KEY)

# Load Embedding Model
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Setup Chat Model
chat_model = genai.GenerativeModel('gemini-1.0-pro-latest')

# Decorator to get function called when POST request sent to /chat
@app.route('/data', methods=['POST'])
def chat():
    # Load input text from json posted
    user_input = request.json.get('data')
    # Error if input is empty
    if not user_input:
        return jsonify({"error": "Empty input text"}), 400
    # Get message from Gemini Pro Chat
    message = gemini_pro_chat_response(user_input, embedding_model, chat_model, collection)
    # Return message as json
    return jsonify({"response": True, "message": message})

# Render the index.html front end
@app.route('/')
def index():
    return render_template('index.html')

# Serve the app with gevent
if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 8080), app)
    http_server.serve_forever()
