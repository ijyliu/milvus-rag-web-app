import streamlit as st

from ollama import Client
client = Client(
  host='http://localhost:3000',
  headers={'Content-Type': 'application/json'}
)

st.title("Chat with Ollama")

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
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
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
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})
