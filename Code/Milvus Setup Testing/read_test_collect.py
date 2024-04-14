from pymilvus import connections, Collection

#curr ids: 448844402818809863, 448844402818809864, 448844402818809865, 448844402818809866, 448844402818809867, 448844402818809868, 448844402818809869, 448844402818809870, 448844402818809871, 448844402818809872

connections.connect(host='localhost', port='19530')
collection_name = 'movies'

collection = Collection(name=collection_name)
query_expression = "description_id in [448844402818809863]"
query_results = collection.query(expr=query_expression, output_fields=["description_id", "vector"])

for result in query_results:
  print(f"ID: {result['description_id']}, Vector: {result['vector']}")
