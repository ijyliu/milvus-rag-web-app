# RAG with Gemini Pro
# Note: Place your Google API key in the Google API Key folder of the directory this code is in, as the contents of a file called 'data-engineering-project.txt'.

##################################################################################################

# Packages
from flask import Flask, request, jsonify, render_template
from RAG_Functions import *
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections
import google.generativeai as genai
import os

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

# Connect to Milvus
connections.connect(host='standalone', port='19530')
collection = Collection("text_embeddings")
collection.load()

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
