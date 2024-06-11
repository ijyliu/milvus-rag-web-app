# app.py
# Milvus RAG web app
# Note: Place your Zilliz token in the Zilliz Token folder of the directory this code is in, as the contents of a file called 'milvus_rag_web_app.txt'.
# Note: Place your Google API key in the Google API Key folder of the directory this code is in, as the contents of a file called 'data-engineering-project.txt'.

##################################################################################################

# Packages
from pymilvus import Collection, MilvusClient, DataType
import pandas as pd
import numpy as np
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from RAG_Functions import *
from sentence_transformers import SentenceTransformer
from pymilvus import Collection
import google.generativeai as genai
import os

##################################################################################################

# Milvus setup

# If collection does not already exist, load embeddings and create index
# Otherwise, load the collection

##################################################################################################

# Load URI from 'Zilliz Token/milvus_rag_web_app.txt'
with open(os.path.expanduser('./Zilliz Token/milvus_rag_web_app.txt')) as f:
    zilliz_uri = f.read().strip()

# Load token from 'Zilliz Token/milvus_rag_web_app.txt'
with open(os.path.expanduser('./Zilliz Token/milvus_rag_web_app.txt')) as f:
    zilliz_token = f.read().strip()

# Create a Milvus client
client = MilvusClient(uri=zilliz_uri, token=zilliz_token)

# Get proposed collection name
collection_name = "text_embeddings"

# Prepare the collection if it does not already exist
if collection_name not in client.list_collections():

    # Prepare the collection if it does not already exist
    schema = client.create_schema(enable_dynamic_field=True, description="Generated text embeddings")

    # Add fields
    schema.add_field(field_name="sentence_id", dtype=DataType.INT64, is_primary=True, auto_id=False)
    schema.add_field(field_name="sentence", dtype=DataType.VARCHAR, max_length=2**15)
    schema.add_field(field_name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="company_name", dtype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="document_name", dtype=DataType.VARCHAR, max_length=64)

    # Create the collection
    client.create_collection(collection_name=collection_name, schema=schema)

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
        # Create list of dictionaries to insert
        records_dicts = []
        for i in range(len(sentences)):

            record = [
                {
                    "sentence": sentences[i],
                    "embedding": embeddings[i],
                    "company_name": company_names[i],
                    "document_name": document_names[i]
                }
            ]

            records_dicts.append(record)
        
        # Insert the data
        _ = client.insert(collection_name=collection_name, records=records_dicts)
    
    # Create Euclidean L2 index
    # Prepare index parameters
    index_params = client.prepare_index_params()
    index_params.add_index(
        metric_type="L2",
        index_type="IVF_FLAT",
        nlist=128
    )
    # Create the index
    client.create_index(collection_name=collection_name, field_name="embedding", index_params=index_params)

    # Load the collection
    client.load_collection(collection_name=collection_name)

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
with open(os.path.expanduser('./Google API Key/data-engineering-project.txt')) as f:
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

# Serving the app on port 5000 when this code is run
if __name__ == '__main__':
    app.run(debug=True, port=5000)
