gcloud run services replace cloudrun.yml --region us-central1
# Limit to max of 2 instances and allow unauthenticated invocations
gcloud run services update milvus-rag-web-app --max-instances 2 --region us-central1
