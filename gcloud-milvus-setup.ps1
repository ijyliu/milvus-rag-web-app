gcloud compute networks create milvus-network --project=milvus-rag-web-app --subnet-mode=auto --mtu=1460 --bgp-routing-mode=regional

gcloud compute firewall-rules create milvus-network-allow-icmp --project=milvus-rag-web-app --network=projects/milvus-rag-web-app/global/networks/milvus-network --description="Allows ICMP connections from any source to any instance on the network." --direction=INGRESS --priority=65534 --source-ranges=0.0.0.0/0 --action=ALLOW --rules=icmp

gcloud compute firewall-rules create milvus-network-allow-internal --project=milvus-rag-web-app --network=projects/milvus-rag-web-app/global/networks/milvus-network --description="Allows connections from any source in the network IP range to any instance on the network using all protocols." --direction=INGRESS --priority=65534 --source-ranges=10.128.0.0/9 --action=ALLOW --rules=all

gcloud compute firewall-rules create milvus-network-allow-rdp --project=milvus-rag-web-app --network=projects/milvus-rag-web-app/global/networks/milvus-network --description="Allows RDP connections from any source to any instance on the network using port 3389." --direction=INGRESS --priority=65534 --source-ranges=0.0.0.0/0 --action=ALLOW --rules=tcp:3389

gcloud compute firewall-rules create milvus-network-allow-ssh --project=milvus-rag-web-app --network=projects/milvus-rag-web-app/global/networks/milvus-network --description="Allows TCP connections from any source to any instance on the network using port 22." --direction=INGRESS --priority=65534 --source-ranges=0.0.0.0/0 --action=ALLOW --rules=tcp:22

gcloud compute firewall-rules create allow-milvus-in --project=milvus-rag-web-app --description="Allow ingress traffic for Milvus on port 19530" --direction=INGRESS --priority=1000 --network=projects/milvus-rag-web-app/global/networks/milvus-network --action=ALLOW --rules=tcp:19530 --source-ranges=0.0.0.0/0
