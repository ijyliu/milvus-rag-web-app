from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os

app = Flask(__name__)

# Load the embedding model
print('Loading Embedding Model...')
model_name = "mixedbread-ai/mxbai-embed-large-v1"
embedding_model = SentenceTransformer(model_name)
print('Embedding Model Loaded.')

def get_mixedbread_of_query(model, query: str):
    '''
    Returns mixedbread embedding for an input text. Text is appropriately formatted to be a query.

    Parameters:
    - model: embedding model
    - query: str: The query to be transformed.
    '''
    transformed_query = f'Represent this sentence for searching relevant passages: {query}'
    return model.encode(transformed_query).tolist() # Convert to list for JSON serialization

@app.route('/encode', methods=['POST'])
def encode_text():
    '''
    Endpoint to get the embedding of a text.
    Expects a JSON payload with a "text" field.
    '''
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" in request body'}), 400

    text = data['text']
    embedding = get_mixedbread_of_query(embedding_model, text)
    return jsonify({'embedding': embedding})

@app.route('/health', methods=['GET'])
def health_check():
    '''
    Health check endpoint to verify if the service is running.
    Returns a simple JSON response.
    '''
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
