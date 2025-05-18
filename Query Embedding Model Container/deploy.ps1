docker build -t us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/query-embedding-model:latest .
docker push us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/query-embedding-model:latest
gcloud run deploy query-embedding-model --image us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/query-embedding-model:latest --region us-central1 --no-allow-unauthenticated --port 5000 --memory 8G --cpu 2 --max-instances 2
