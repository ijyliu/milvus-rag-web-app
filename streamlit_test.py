import streamlit as st
import requests
import json

st.title("Ollama Stream in Streamlit")

prompt = st.text_input("Enter your prompt")

if prompt:
    response = requests.post(
        "http://localhost:3000/api/generate",
        json={"model": "gemma3:1b", "prompt": prompt, "stream": True},
        stream=True,
    )

    output_placeholder = st.empty()
    full_output = ""

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            full_output += data.get("response", "")
            output_placeholder.markdown(full_output)
