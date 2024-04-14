# query testing (of text company_name though)
from pymilvus import connections, Collection

connections.connect(host='localhost', port='19530')

collection_name = 'text_embeddings'
collection = Collection(name=collection_name)
collection.load()

expr = f"company_name == 'TheHersheyCompany'"

results = collection.query(expr=expr)

for result in results:
    print(result)

    