# RAG_Fuctions

###################################################################################################

# Packages
import time
import requests
import json

###################################################################################################

def get_mixedbread_of_query(query, embedding_model_url):
    '''
    Returns mixedbread embedding for an input text. Text is appropriately formatted to be a query.

    Parameters:
    - query: str: The query to be transformed.
    - embedding_model_url: str: The URL of the embedding model API.
    '''
    # Required format for query
    transformed_query = f'Represent this sentence for searching relevant passages: {query}'
    # Set headers and data for the request
    headers = {'Content-Type': 'application/json'}
    data = {"model": "mxbai-embed-large:335m",
            "prompt": transformed_query}
    # Get embedding
    res = requests.post(
        url=embedding_model_url,
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

def construct_prompt(input_text, collection, embedding_model_url):
    '''
    Constructs a prompt for the Gemma model based on the user input and the top 5 sentences from the Milvus collection.

    Parameters:
    - input_text: str: The user query.
    - collection: Milvus collection.
    - embedding_model_url: str: The URL of the embedding model API.
    '''
    
    # Get embedding of input
    input_embedding = get_mixedbread_of_query(input_text, embedding_model_url)
    print('got embedding')

    # Top5 sentences
    top5_sentences, documents_cited, milvus_query_time = return_top_5_sentences(collection, input_embedding)

    # Construct prompt
    prompt_lines = ["Context That May Be Helpful (You May Disregard if Not Helpful):"] + top5_sentences + ["User Query:\n" + input_text]
    prompt = "\n".join(prompt_lines)

    # Return prompt and metadata
    return prompt, documents_cited, milvus_query_time
