docker stop $(docker ps -a -q --filter ancestor=us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/query-embedding-model:latest)
docker stop $(docker ps -a -q --filter ancestor=us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/ollama-gemma:latest)
docker stop $(docker ps -a -q --filter ancestor=us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/main-app:latest)
docker rm $(docker ps -a -q --filter ancestor=us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/query-embedding-model:latest)
docker rm $(docker ps -a -q --filter ancestor=us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/ollama-gemma:latest)
docker rm $(docker ps -a -q --filter ancestor=us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/main-app:latest)
Start-Job -ScriptBlock { docker run -p 5000:5000 us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/query-embedding-model:latest }
Start-Job -ScriptBlock { docker run -p 3000:3000 us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/ollama-gemma:latest }
Start-Job -ScriptBlock { docker run -p 8080:8080 us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/main-app:latest }
