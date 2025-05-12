docker build -t gcr.io/milvus-rag-web-app/ollama-gemma:latest .
docker push gcr.io/milvus-rag-web-app/ollama-gemma:latest
gcloud run deploy ollama-gemma --image gcr.io/milvus-rag-web-app/ollama-gemma:latest --region us-central1 --no-allow-unauthenticated --port 3000 --memory 8G --cpu 2 --max-instances 2
