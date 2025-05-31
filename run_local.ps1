docker run -p 5000:5000 us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/query-embedding-model:latest
docker run -p 3000:3000 us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/ollama-gemma:latest
docker run -p 8080:8080 us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/main-app:latest
