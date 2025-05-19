# RAG_Fuctions

###################################################################################################

# Packages
import time
import requests
import json

###################################################################################################

def get_mixedbread_of_query(query):
    '''
    Returns mixedbread embedding for an input text. Text is appropriately formatted to be a query.

    Parameters:
    - query: str: The query to be transformed.
    '''
    # Required format for query
    transformed_query = f'Represent this sentence for searching relevant passages: {query}'
    # Set headers and data for the request
    headers = {'Content-Type': 'application/json'}
    data = {"model": "mxbai-embed-large:335m",
            "prompt": transformed_query}
    # Get embedding
    res = requests.post(
        url='http://localhost:5000/api/embeddings',
        headers=headers,
        data=json.dumps(data)
    )
    # Return embedding
    return res.json()['embedding']

def return_top_5_sentences(collection, query_embedding):
    '''
    Returns top 5 sentences from the collection based on the query embedding. Also includes unique associated files used and time taken.

    Parameters:
    - collection: Milvus collection
    - query_embedding: The embedding of the query text.
    '''

    # Set search parameters
    search_params = {
        "metric_type": "L2", # similarity metric
        "offset": 0, # the number of top-k hits to skip
        "ignore_growing": False # do not ignore growing segments
    }

    # Start timer
    start_time = time.time()

    # Use Milvus to search for similar vectors
    results = collection.search(
        data=[query_embedding], # query vector
        anns_field="embedding", # name of field to search on
        param=search_params, # seach parameters set above
        limit=5,# num results to return
        expr=None, # boolean filter
        output_fields=['company_name', 'sentence', 'document_name'], # fields to return 
        consistency_level="Strong"
    )

    # End timer
    end_time = time.time()

    # Get sentences, companies, and documents from results
    sentences = []
    companies = []
    documents = []
    for hits in results:
        for hit in hits:
            sentences.append(hit.get("sentence"))
            companies.append(hit.get("company_name"))
            documents.append(hit.get("document_name"))

    # Get filenames
    # Join company and document names on underscore and add .txt
    filenames = [f'{company}_{document}.txt' for company, document in zip(companies, documents)]
    # Keep unique values
    filenames = list(set(filenames))

    # Return sentences, filenames, and time taken
    return sentences, filenames, end_time - start_time

def send_to_gemma(prompt):
    '''
    Sends a prompt to the Gemma model and returns the response.

    Parameters:
    - prompt: str: The prompt to be sent to the model.
    '''

    print('sending to gemma')
    
    # Data and headers setup
    headers = {'Content-Type': 'application/json'}
    data = {"model": "gemma3:1b", "prompt": prompt}

    # Make request
    try:
        response = requests.post(url='http://localhost:3000/api/generate',
                                 headers=headers, 
                                 data=json.dumps(data), 
                                 stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    # Convert stream to text
    full_response = ""
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data = json.loads(line)
                if "response" in json_data:
                    full_response += json_data["response"]
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} - Line: {line}")

    # print("Full LLM Response:")
    # print(full_response)

    return full_response

def gemma_chat_response(input_text, collection):
    '''
    Chat with the Gemma model. Returns the response of the model to the user query.

    Parameters:
    - input_text: str: The user query.
    - collection: Milvus collection.
    '''
    
    # Get embedding of input
    input_embedding = get_mixedbread_of_query(input_text)
    print('got embedding')

    # Top5 sentences
    top5_sentences, documents_cited, milvus_query_time = return_top_5_sentences(collection, input_embedding)

    # Construct prompt
    prompt_lines = ["Context That May Be Helpful (You May Disregard if Not Helpful):"] + top5_sentences + ["User Query:\n" + input_text]
    prompt = "\n".join(prompt_lines)

    # Get response
    # Start timer
    start_time = time.time()
    chat_response = send_to_gemma(prompt)
    # End timer
    end_time = time.time()
    # Chat model response time
    chat_model_response_time = end_time - start_time

    # Format response for user
    response_for_user = "Assistant: " + chat_response + "\n\nDocuments Cited: " + ', '.join(documents_cited) + "\n\nMilvus Query Time: " + str(round(milvus_query_time, 2)) + ' seconds' + "\n\nChat Model Response Time: " + str(round(chat_model_response_time, 2)) + ' seconds'

    # Return response
    return response_for_user
