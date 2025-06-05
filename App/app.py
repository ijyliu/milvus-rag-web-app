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
import pandas as pd

##################################################################################################

# Read environment variable "ENVIRONMENT"
environment = os.getenv("ENVIRONMENT", "non-production")
if environment == "production":
    chat_model_url = "https://localhost:3000"
    embedding_model_url = "https://localhost:5000/api/embeddings"
else:
    chat_model_url = "http://host.docker.internal:3000"
    embedding_model_url = "http://host.docker.internal:5000/api/embeddings"

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

# Set up Ollama client
client = Client(
  host=chat_model_url,
  headers={'Content-Type': 'application/json'}
)

# Set page configs
st.set_page_config(
    page_title="Terms of Service Chatbot",
    page_icon=":page_with_curl:",
)
# Title and description
st.title("Terms of Service Chatbot")
st.markdown('''
This app allows you to chat with an AI assistant about nearly 2,000 online terms of service documents. You can ask questions, and the assistant will provide answers based on the context of the documents.
            
You can search the table below to get a sense of what documents might be drawn upon. Upon mouseover, a search button will appear on the upper right.
''')

# Pandas dataframe in doc_df.csv
doc_df = pd.read_csv(os.path.expanduser('doc_df.csv'))
st.dataframe(
             data = doc_df,
             height = 150,
             hide_index=True
             )

st.markdown('''
Chats do not persist between sessions, but you are welcome to print the page to keep a record of your conversation.
''')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are an expert assistant that converses with users concerning online terms of service documents. You are able to draw on context of specific sentences retrieved from these documents in your responses. You may disregard some or all of the context if it is not helpful. Please note the user does provide you with the context or know what it says - it has just been attached to their query."}
    ]

# Display chat input
user_input = st.chat_input("Example: What does Apple do to protect my privacy on iCloud?")

# Display existing chat history
for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            content_to_display = message["content"]
            # If role is user, split on 'User Query:' and display items after that
            if message["role"] == "user":
                content_to_display = content_to_display.split('User Query:')[-1].strip()
            # Display the content
            st.write(content_to_display)

# Receive input
if user_input:

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Brief loading animation
    # Set columns for spinner
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:

        # Spinner with text
        with st.spinner("Rewriting user input and running retrieval..."):

            # Input rewriting
            # Version of conversation history with User Query as their messages only (basically user-facing conversation history)
            uf_conversation_history = []
            for message in st.session_state.messages:
                if message["role"] == "user":
                    uf_conversation_history.append({"role": "user", "content": message["content"].split('User Query:')[-1].strip()})
                else:
                    uf_conversation_history.append({"role": message["role"], "content": message["content"]})
            rewritten_input, _ = rewrite_user_input(uf_conversation_history, user_input, chat_model_url)
            print('rewritten_input:', rewritten_input)
            
            # Run RAG to add context to the user input
            prompt, documents_cited, milvus_query_time = construct_prompt(rewritten_input, collection, embedding_model_url)
            # Swap prompt back to making use of the original user input
            prompt = prompt.split('User Query:')[0] + 'User Query: ' + user_input.strip()
            print('prompt:', prompt)

            # Append user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get and display streaming response
    with st.chat_message("assistant"):

        # Display a placeholder for the message
        message_placeholder = st.empty()

        # Initialize full response
        full_response = ""
        
        # Set up completion streaming
        completion = client.chat(
            model="gemma3:1b",
            messages=st.session_state.messages,
            stream=True
        )

        # Chunk streaming
        for chunk in completion:

            # Writing streaming message content
            if 'message' in chunk and 'content' in chunk['message']:
                content = chunk['message']['content']
                full_response += content
                message_placeholder.write(full_response + "▌")

        # Write final response with documents cited
        message_placeholder.write(full_response + "\n\n---\n\nDocuments Cited: " + ', '.join(documents_cited))

    # Append the full response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response + "\n\n---\n\nDocuments Cited: " + ', '.join(documents_cited)})
