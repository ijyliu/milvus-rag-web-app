from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections, utility
import json

file_path = 'dummy_data.json'

with open(file_path, 'r') as json_file:
  description_vectors_list = json.load(json_file)

connections.connect(host='localhost', port='19530')
collection_name = 'movies'

if collection_name not in utility.list_collections():
  fields = [
    FieldSchema(name="description_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=128)
  ]
  
  schema = CollectionSchema(fields, description="Description Embeddings")
  collection = Collection(name=collection_name, schema=schema)
  print(f"Collection {collection_name} created.")
else:
  collection = Collection(name=collection_name)
  print(f"Collection {collection_name} already exists.")

vectors = [entry["vector"] for entry in description_vectors_list]
insert_result = collection.insert([vectors])
print("Data inserted. IDs:", insert_result.primary_keys)

collection.create_index(field_name="vector", index_params={"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
collection.load()
