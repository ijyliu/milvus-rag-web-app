FROM ollama/ollama:latest

# Listen on all interfaces, port 3000
ENV OLLAMA_HOST 0.0.0.0:3000

# Store model weight files in /models
ENV OLLAMA_MODELS /models

# Reduce logging verbosity
ENV OLLAMA_DEBUG true

# Never unload model weights
ENV OLLAMA_KEEP_ALIVE -1

# Store the model weights in the container image
ENV MODEL gemma3:1b
RUN ollama serve & sleep 5 && ollama pull $MODEL

# Start Ollama
ENTRYPOINT ["ollama", "serve"]
