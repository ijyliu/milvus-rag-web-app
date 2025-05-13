docker build -t us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/ollama-gemma:latest .
docker push us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/ollama-gemma:latest
gcloud run deploy ollama-gemma --image us-central1-docker.pkg.dev/milvus-rag-web-app/milvus-rag-web-app/ollama-gemma:latest --region us-central1 --no-allow-unauthenticated --port 3000 --memory 8G --cpu 2 --max-instances 2
