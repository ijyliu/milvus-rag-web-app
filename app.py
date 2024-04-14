# app.py
# Milvus RAG web app
# Note: Place your Google API key in the Google API Key folder of the directory this code is in, as the contents of a file called 'data-engineering-project.txt'.

##################################################################################################

# Packages
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import pandas as pd
import numpy as np
import os
import time
from flask import Flask, request, jsonify, render_template
from RAG_Functions import *
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
import google.generativeai as genai
import os

##################################################################################################

# If collection does not already exist, load embeddings and create index
# Otherwise, load the collection

##################################################################################################

connections.connect("default", host="standalone", port="19530")
collection_name = "text_embeddings"

# Prepare the collection if it does not already exist
if collection_name not in utility.list_collections():

    # fields: entences, embedding, companies, and documents
    fields = [
        FieldSchema(name="sentence_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        # we got some long sentences in here so length for this field has to be quite long to accomodate some outliers
        FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=2**15),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=64),
    ]

    schema = CollectionSchema(fields, description="Generated text embeddings")

    collection = Collection(name=collection_name, schema=schema)

    dir_path = './Embeddings'

    t0 = time.time()
    for path in os.listdir(dir_path):
        curr_company_name = path.split('_')[0]
        curr_document_name = path.split('_')[1].split('.')[0]

        df = pd.read_parquet(f'{dir_path}/{path}')

        sentences = df['sentence'].to_list()
        embeddings = df.filter(regex='^embed_element_').values.tolist()

        company_names = np.full(len(sentences), curr_company_name)
        document_names = np.full(len(sentences), curr_document_name)

        if len(curr_company_name) > 64:
            print(curr_company_name)
        if len(curr_document_name) > 64:
            print(curr_document_name)
        
        mr = collection.insert(
            [
                sentences,      # sentences
                embeddings,     # embeddings
                company_names,  # company names
                document_names, # document names
            ]
        )
    
    # Create Euclidean L2 index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()
# Otherwise, load the collection
else:
    collection = Collection(name=collection_name)
    # Check if index exists
    if not collection.has_index():
        # Create Euclidean L2 index
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128}
        }
        collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

##################################################################################################

# Running the app

##################################################################################################

# Create flask web server
app = Flask(__name__)

# Load API key
with open(os.path.expanduser('./Google API Key/data-engineering-project.txt')) as f:
    GOOGLE_API_KEY = f.read().strip()
genai.configure(api_key=GOOGLE_API_KEY)

# Load Embedding Model
embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Setup Chat Model
chat_model = genai.GenerativeModel('gemini-1.0-pro-latest')

# Decorator to get function called when user sends POST request to /chat
@app.route('/chat', methods=['POST'])
def chat():
    # Load input text from json posted
    user_input = request.json.get('input_text')
    # Error if input is empty
    if not user_input:
        return jsonify({"error": "Empty input text"}), 400
    # Get response from Gemini Pro Chat
    response = gemini_pro_chat_response(user_input, embedding_model, chat_model, collection)
    # Return response as json
    return jsonify({"response": response})

# Render the index.html front end
@app.route('/')
def index():
    return render_template('index.html')

# Serving the app on port 5000 when this code is run
if __name__ == '__main__':
    app.run(debug=True, port=5000)
