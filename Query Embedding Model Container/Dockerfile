FROM ollama/ollama:0.7.0

# Listen on all interfaces, port 5000
ENV OLLAMA_HOST 0.0.0.0:5000

# Store model weight files in /models
ENV OLLAMA_MODELS /models

# Reduce logging verbosity
ENV OLLAMA_DEBUG false

# Never unload model weights
ENV OLLAMA_KEEP_ALIVE -1

# Store the model weights in the container image
ENV MODEL mxbai-embed-large:335m
RUN ollama serve & sleep 5 && ollama pull $MODEL

# Start Ollama
ENTRYPOINT ["ollama", "serve"]
