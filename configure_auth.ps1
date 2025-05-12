$filePath = "Credentials/service_account.txt"
$quotedString = Get-Content -Path $filePath
$serviceAccount = $quotedString.Trim()

echo "Service Account: $serviceAccount"

gcloud run services add-iam-policy-binding ollama-gemma --member=$serviceAccount --role="roles/run.invoker" --region=us-central1

gcloud run services add-iam-policy-binding query-embedding-model --member=$serviceAccount --role="roles/run.invoker" --region=us-central1
