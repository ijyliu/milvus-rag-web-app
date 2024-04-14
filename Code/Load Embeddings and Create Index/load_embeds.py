# run me to load milvus embeddings! if your db is messed up re run me and i will fix it :)

from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import pandas as pd
import numpy as np
import os
import time

connections.connect("default", host="localhost", port="19530")
collection_name = "text_embeddings"

#if the collection excists then drop it
if collection_name in utility.list_collections():
    collection = Collection(name=collection_name)
    collection.drop() 

#these are the fields I want, sentences, embedding, companies, and documents
fields = [
    FieldSchema(name="sentence_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    #we got some long sentences in here so length for this field has to be quite long to accomodate some outliers
    FieldSchema(name="sentence", dtype=DataType.VARCHAR, max_length=2**15),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name="company_name", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="document_name", dtype=DataType.VARCHAR, max_length=64),
]

schema = CollectionSchema(fields, description="Generated text embeddings")

collection = Collection(name=collection_name, schema=schema)

dir_path = '../../Embeddings'

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
    print(f"Inserted for {curr_company_name}: {curr_document_name}")

print(f"insertion completed in {time.time() - t0} seconds")
