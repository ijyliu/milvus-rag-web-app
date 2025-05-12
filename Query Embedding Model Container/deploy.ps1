docker build -t gcr.io/milvus-rag-web-app/query-embedding-model:latest .
docker push gcr.io/milvus-rag-web-app/query-embedding-model:latest
gcloud run deploy query-embedding-model --image gcr.io/milvus-rag-web-app/query-embedding-model:latest --region us-central1 --no-allow-unauthenticated --port 5000 --memory 8G --cpu 2 --max-instances 2
