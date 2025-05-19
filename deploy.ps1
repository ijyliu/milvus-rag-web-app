gcloud run services replace cloudrun.yml --region us-central1
# Allow unauthenticated invocations
gcloud run services add-iam-policy-binding milvus-rag-web-app --region='us-central1' --member='allUsers' --role='roles/run.invoker' --project='milvus-rag-web-app'
