# mixedbread_api_test.py
# Compare mixedbread API embeddings with sentence-transformer embeddings

##################################################################################################

# Packages
from mixedbread_ai.client import MixedbreadAI
import os
from sentence_transformers import SentenceTransformer

##################################################################################################

# Test query
query = 'What does Apple do to protect my data?'
# Required format for query
transformed_query = f'Represent this sentence for searching relevant passages: {query}'

##################################################################################################

# API Call

# Load mixedbread API key
with open(os.path.expanduser('Credentials/mixedbread_api_key.txt')) as f:
    MIXEDBREAD_API_KEY = f.read().strip()

# Setup MixedbreadAI client
mxbai_client = MixedbreadAI(api_key=MIXEDBREAD_API_KEY)

# Get embedding
res = mxbai_client.embeddings(
    model='mixedbread-ai/mxbai-embed-large-v1',
    input=transformed_query,
    dimensions=1024,
    normalized=False
)
# Return embedding
print(res.data[0].embedding[:5])

##################################################################################################

# Local sentence transformer embeddings

embedding_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

embedding_model.encode(transformed_query)
print(embedding_model.encode(transformed_query)[:5])
