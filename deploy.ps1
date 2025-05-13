docker build -t us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/milvus-rag-web-app:latest .
docker push us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/milvus-rag-web-app:latest
gcloud run deploy milvus-rag-web-app --image=us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/milvus-rag-web-app:latest --region=us-central1 --allow-unauthenticated --max-instances=2
