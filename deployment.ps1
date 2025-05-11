docker build -t gcr.io/milvus-rag-web-app/milvus-rag-web-app:latest .
docker push gcr.io/milvus-rag-web-app/milvus-rag-web-app:latest
gcloud run deploy milvus-rag-web-app --image=gcr.io/milvus-rag-web-app/milvus-rag-web-app:latest --region=us-central1 --allow-unauthenticated --max-instances=2
