# Create L2 (Euclidean distance) index for the collection.

from pymilvus import connections, Collection

connections.connect(host='localhost', port='19530')

collection_name = 'text_embeddings'
collection = Collection(name=collection_name)
# Create Euclidean L2 index
index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
collection.load()
