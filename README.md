# Milvus RAG Web App

A Retrieval-Augmented-Generation chatbot that draws upon the TOSDR (Terms of Service Didn't Read) corpus to answer your questions about online privacy and more

![image](https://github.com/user-attachments/assets/ca80583f-990a-4cd4-9917-417cec851f16)

You can interact with the bot [here](https://milvus-rag-web-app-264286705270.us-central1.run.app/).

To construct this app, each sentence in the TOSDR corpus (with the company name inserted) was converted to an embedding using Mixedbread.ai's state-of-the-art embedding model, and loaded into Milvus with a Euclidean IVF Flat index (L2 distance, clustered nearest neighbor search). When the user asks a question it is encoded with the model, and the top 5 most similar vectors are fetched from Milvus. The sentences for these vectors are appended to the prompt context for a deployed version of Google Gemma, which draws upon the information to answer the user's query.

## Technologies (not exhaustive!)

- Milvus Vector DB through Zilliz Cloud
- Python
  - PyMilvus SDK
  - Streamlit
  - Flask (no longer used)
- Ollama
- Streamlit
- Docker
- Google Cloud Platform

## Acknowledgements

Based on [data engineering work](https://github.com/ijyliu/data-engineering-project) completed with Fangyuan Li and Robert Thompson
