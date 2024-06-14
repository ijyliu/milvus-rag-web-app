# Load_Embeddings.py
# Fills a Milvus collection with embeddings from a directory of parquet files.
# This file should be run from the root of the repository.
# Put a 'Credentials' folder in the root of this repository with the following files:
# Note: Place your Zilliz URI in a file called 'zilliz_uri.txt'.
# Note: Place your Zilliz token in a file called 'zilliz_token.txt'.

##################################################################################################

# Packages
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType
import pandas as pd
import numpy as np
import os

##################################################################################################

# Local flag
# Changes connection settings to a local Milvus instance
local = False

##################################################################################################

# Milvus setup

# Load URI from './Credentials/zilliz_uri.txt'
with open(os.path.expanduser('./Credentials/zilliz_uri.txt')) as f:
    zilliz_uri = f.read().strip()

# Load token from './Credentials/zilliz_token.txt'
with open(os.path.expanduser('./Credentials/zilliz_token.txt')) as f:
    zilliz_token = f.read().strip()

# Connect to Milvus offered via Zilliz
connections.connect(alias="default", uri=zilliz_uri, token=zilliz_token)

# Set up collection name
collection_name = "text_embeddings"

##################################################################################################

# Load the embeddings into the collection

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
dir_path = './Load Embeddings/Embeddings'

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
