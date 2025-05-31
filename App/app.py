# app.py
# Milvus RAG web app
# Put a 'Credentials' folder in the root of this repository
# Place your Zilliz URI in a file called 'zilliz_uri.txt'
# Place your Zilliz token in a file called 'zilliz_token.txt'

##################################################################################################

# Packages
from pymilvus import Collection, connections
import os
from RAG_Functions import *
import streamlit as st
from ollama import Client

##################################################################################################

# Read environment variable "ENVIRONMENT"
environment = os.getenv("ENVIRONMENT", "local")
# Set up URLs for chat and embedding models based on the environment
if environment == "local":
    chat_model_url = "http://host.docker.internal:3000"
    embedding_model_url = "http://host.docker.internal:5000/api/embeddings"
elif environment == "production":
    chat_model_url = "https://localhost:3000"
    embedding_model_url = "https://localhost:5000/api/embeddings"

##################################################################################################

# Flag to change connection settings to a local Milvus instance
local_milvus = False

##################################################################################################

# Milvus setup

# Load URI from '.../Credentials/zilliz_uri.txt'
with open(os.path.expanduser('../Credentials/zilliz_uri.txt')) as f:
    zilliz_uri = f.read().strip()

# Load token from '.../Credentials/zilliz_token.txt'
with open(os.path.expanduser('../Credentials/zilliz_token.txt')) as f:
    zilliz_token = f.read().strip()

# Connect to Milvus
if local_milvus:
    connections.connect("default", host="localhost", port="19530")
if not local_milvus:
    connections.connect(alias="default", uri=zilliz_uri, token=zilliz_token)

# Set up collection name
collection_name = "text_embeddings"

print('set up collection name')

##################################################################################################

# Load the collection and create index if it does not exist

# Get collection
collection = Collection(name=collection_name)

# Check if index exists, create if it does not
if not collection.has_index():
    # Create Euclidean L2 index
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

# Load the collection
collection.load()

print('loaded collection')

##################################################################################################

# Running the app

##################################################################################################

client = Client(
  host=chat_model_url,
  headers={'Content-Type': 'application/json'}
)

# Set page configs
st.set_page_config(
    page_title="Terms of Service Chatbot",
    page_icon=":page_with_curl:",
)
st.title("Terms of Service Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

# Display chat input
user_input = st.chat_input("Your message:")

# Display existing chat history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

if user_input:
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Run RAG to add context to the user input
    prompt, documents_cited, milvus_query_time = construct_prompt(user_input, collection, embedding_model_url)

    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display streaming response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        completion = client.chat(
            model="gemma3:1b",
            messages=st.session_state.messages,
            stream=True
        )
        for chunk in completion:
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                full_response += content
                message_placeholder.write(full_response + "▌")
        message_placeholder.write(full_response)
    
    # # Format response for user
    # response_for_user = "Assistant: " + chat_response + "\n\nDocuments Cited: " + ', '.join(documents_cited) + "\n\nMilvus Query Time: " + str(round(milvus_query_time, 2)) + ' seconds' + "\n\nChat Model Response Time: " + str(round(chat_model_response_time, 2)) + ' seconds'

    # Append the full response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
