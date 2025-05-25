import streamlit as st
import requests
import json
import time
import os

model = "gemma3:1b"

def show_msgs():
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            st.write(msg["content"])

def chat(messages, model="gemma3:1b"):
    try:
        response = requests.post(
            "http://localhost:3000/api/chat",
            json={"model": model, "messages": messages, "stream": True},
            stream=True
        )
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if line:
                try:
                    body = json.loads(line)
                    if "error" in body:
                        raise Exception(body["error"])
                    if "message" in body and "content" in body["message"]:
                        yield body["message"]["content"]
                    if body.get("done", False):
                        break
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                except Exception as e:
                    yield str(e)
                    break
    except requests.exceptions.RequestException as e:
        yield f"Request error: {e}"
    except Exception as e:
        yield f"An unexpected error occurred: {e}"

def format_messages_for_summary(messages):
    return '\n'.join(f"{msg['role']}: {msg['content']}" for msg in messages)

def summary(messages):
    sysmessage = "summarize this conversation in 3 words. No symbols or punctuation:\n\n\n"
    combined = sysmessage + messages
    api_message = [{"role": "user", "content": combined}]
    try:
        response = requests.post(
            "http://localhost:3000/api/chat",
            json={"model": model, "messages": api_message, "stream": True},
        )
        response.raise_for_status()
        output = ""
        for line in response.iter_lines():
            body = json.loads(line)
            if "error" in body:
                raise Exception(body["error"])
            if body.get("done", False):
                return output
            output += body.get("message", {}).get("content", "")
    except Exception as e:
        return str(e)

def save_chat():
    if not os.path.exists('./Intermediate-Chats'):
        os.makedirs('./Intermediate-Chats')
    if st.session_state['messages']:
        formatted_messages = format_messages_for_summary(st.session_state['messages'])
        chat_summary = summary(formatted_messages)
        filename = f'./Intermediate-Chats/{chat_summary}.txt'
        with open(filename, 'w') as f:
            for message in st.session_state['messages']:
                encoded_content = message['content'].replace('\n', '\\n')
                f.write(f"{message['role']}: {encoded_content}\n")
        st.session_state['messages'].clear()
    else:
        st.warning("No chat messages to save.")

def load_saved_chats():
    chat_dir = './Intermediate-Chats'
    if os.path.exists(chat_dir):
        files = os.listdir(chat_dir)
        files.sort(key=lambda x: os.path.getmtime(os.path.join(chat_dir, x)), reverse=True)
        for file_name in files:
            display_name = file_name[:-4] if file_name.endswith('.txt') else file_name
            if st.sidebar.button(display_name):
                st.session_state['show_chats'] = False
                st.session_state['is_loaded'] = True
                load_chat(f"./Intermediate-Chats/{file_name}")

def format_chatlog(chatlog):
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in chatlog)

def load_chat(file_path):
    st.session_state['messages'].clear()
    show_msgs()
    with open(file_path, 'r') as file:
        for line in file.readlines():
            role, content = line.strip().split(': ', 1)
            decoded_content = content.replace('\\n', '\n')
            st.session_state['messages'].append({'role': role, 'content': decoded_content})

def main():
    st.title("LLaMA Chat Interface")
    user_input = st.chat_input("Enter your prompt:", key="1")
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    show_msgs()
    if user_input:
        with st.chat_message("user",):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        messages = [{"role": "user", "content": "\\n".join(msg["content"] for msg in st.session_state.messages if msg["role"] == "user")}]
        with st.chat_message("assistant"):
            response_stream = chat(messages)
            placeholder = st.empty()
            full_response = ""
            for chunk in response_stream:
                full_response += chunk
                placeholder.markdown(full_response + "▌")
                time.sleep(0.05)  # Small delay to help Streamlit update
                st.experimental_rerun() # Force Streamlit to re-run and update
            placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    elif st.session_state['messages'] is None:
        st.info("Enter a prompt or load chat above to start the conversation")
    chatlog = format_chatlog(st.session_state['messages'])
    st.sidebar.download_button(
        label="Download Chat Log",
        data=chatlog,
        file_name="chat_log.txt",
        mime="text/plain"
    )
    for i in range(5):
        st.sidebar.write("")
    if st.sidebar.button("Save Chat"):
        save_chat()

    # Show/Hide chats toggle
    if st.sidebar.checkbox("Show/hide chat history", value=st.session_state['show_chats']):
        st.sidebar.title("Previous Chats")
        load_saved_chats()

    for i in range(3):
        st.sidebar.write(" ")

if __name__ == "__main__":
    main()
