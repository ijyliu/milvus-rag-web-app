from pymilvus import connections, utility

connections.connect(host='localhost', port='19530')
collections = utility.list_collections()
print(f'All collections: {collections}')