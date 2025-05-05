FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY . .

# Real-time logging
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["python", "Query_Embedding_Server.py"]
