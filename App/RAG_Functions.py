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

def stringify_conversation_history(conversation_history):
    '''
    Converts the conversation history into a string format suitable for use in prompts.

    Parameters:
    - conversation_history: list: The conversation history to be converted.
    '''
    # Convert each message in the conversation history to a string
    conversation_history_str = "\n".join(
        [f"{entry['role']}: {" ".join(entry['content'].split())}" for entry in conversation_history if entry['role'] != 'system']
    )
    return conversation_history_str

def rewrite_user_input(conversation_history, input_text, chat_model_url):
    '''
    Rewrites the user's query or input to be more suitable for the embedding model.

    Parameters:
    - conversation_history: list: The conversation history to provide context.
    - input_text: str: The user's query.
    - chat_model_url: str: The URL of the chat model API.
    '''

    # Generate endpoint URL for the chat model
    generate_endpoint = chat_model_url + '/api/generate'

    # Examples to use in prompt
    example_1_conv_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What does Apple do to protect my privacy on iCloud?"},
        {"role": "assistant", "content": '''
            Okay, let's break down what Apple does to protect your privacy on iCloud. Based on the provided text, here’s a summary of their key practices:

            Apple employs several measures to protect your privacy on iCloud:

            End-to-End Encryption: They use this to protect sensitive information – meaning only you have the key to decrypt your data.
            Account Security: They’re committed to safeguarding your account itself.
            Data Security: They prioritize data security across all their systems.
            Responsible Data Handling: Apple believes strong privacy is linked to strong security.
            Essentially, Apple uses a multi-layered approach to minimize the risk of data breaches and unauthorized access.

            Do you want me to elaborate on any of these points, or perhaps answer a follow-up question about their privacy practices? For example, would you like me to:

            Explain end-to-end encryption in more detail?
            Discuss how they handle account security?
        '''
        }
    ]
    example_1_input_text = "Do they share my data with third parties?"
    example_1_rewritten_input = "Does Apple share my data with third parties?"
    example_2_conv_history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    example_2_input_text = "What is the best way to protect my data on iCloud?"
    example_2_rewritten_input = "What is the best way to protect my data on Apple's iCloud?"
    # example_3_conv_history = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "What am I allowed to post on Facebook?"},
    #     {"role": "assistant", "content": '''
    #         Okay, let’s tackle that question! Here’s a breakdown of what you’re allowed to post on Facebook, based on the information you’ve provided:

    #         Generally, you have considerable control over what you share. Facebook allows you to choose who sees your posts.

    #         Here’s a breakdown of what’s generally permitted:

    #         Personal Content: You can share your thoughts, experiences, photos, videos, and updates about your life. This includes anything you want to be public.
    #         User-Generated Content: You’re allowed to post content created by other people – like posts from friends, family, or people you follow.
    #         However, there are some important caveats and restrictions:

    #         Third-Party Rights: You agree that your content doesn’t violate the rights of any third party. This means you need to be careful about copyright, trademark, and other intellectual property laws.
    #         Legal Reasons: Facebook has the right to remove or block content that violates the law.
    #         In short, you can share pretty much anything you want, but be mindful of potential legal issues and the rights of others.

    #         To give you more specific advice, could you tell me what kind of content you're interested in? For example, are you wondering about sharing photos, videos, or just general updates?
    #     '''
    #     },
    #     {"role": "user", "content": "I'm interested in what kinds of videos I'm allowed to post on Facebook."},
    #     {"role": "assistant", "content": '''
    #         Okay, let’s break down what’s allowed on Facebook regarding videos – based on the context you’ve provided:

    #         Generally, you can post videos on Facebook, but with some restrictions. Facebook has a policy to prevent the upload of videos that violate their guidelines. Here’s a breakdown:

    #         Content Restrictions: You cannot post videos that are inappropriate, profane, offensive, false, disparaging, defamatory, obscene, illegal, infringing, threatening, or sexually explicit.
    #         “Inappropriate, Profane, Vulgar, Offensive, False, Disparaging, Defamatory, Illegal, Infringing, Threatening” – These are key areas to be mindful of.
    #         Important Clarifications:

    #         “We deem, in our sole discretion” – This is a crucial phrase. Facebook has the final say on what content is prohibited. They can decide if a video violates their policy based on their own judgment.
    #         Embedding Restrictions: If you embed a video on any website, you must keep it off that website. Think of it as a 'walled' environment.
    #         Essentially, Facebook wants to ensure the videos you share are appropriate and don’t violate their terms of service.

    #         To help me give you a more tailored answer, could you tell me:

    #         Are you thinking about sharing any type of video?
    #     '''
    #     }
    # ]
    # example_3_input_text = "Clips from movies."
    # example_3_rewritten_input = "Can I post clips from movies on Facebook?"
    example_4_conv_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Are disputes concerning Amazon kindle subject to arbitration?"},
        {"role": "assistant", "content": '''
            Okay, let’s dive into the context of Amazon disputes and binding arbitration regarding Kindle books:

            Yes, disputes concerning Amazon Kindle books are subject to binding arbitration.

            Here’s a breakdown based on the provided context:

            4.5 Disputes / Binding Arbitration: This is a significant indicator. It means that Amazon has established a process for resolving disputes with Kindle users.
            5.5 Disputes / Binding Arbitration: This reinforces the established process.
            4.4 Disputes / Binding Arbitration: This further confirms the established process.
            In short, Amazon uses binding arbitration to handle disputes with Kindle users.

            Do you have any other questions about this process or Amazon’s dispute resolution?
        '''
        }
    ]
    example_4_input_text = "What about for Netflix?"
    example_4_rewritten_input = "Are disputes concerning Netflix subject to arbitration?"

    # Prepare the prompt for the chat model
    rewrite_prompt = f'''
    Rewrite the following user input to be more suitable for an embedding model. The rewritten input should be self contained and incorporate conversation history as necessary. For example, if no company name or question topic is specified in the user input, you should add that information based on the conversation history. However, take care to preserve the meaning of the original input. Your rewritten input should be only one sentence or question.

    EXAMPLES:

    Conversation history: {stringify_conversation_history(example_1_conv_history)}
    
    User input to rewrite: {example_1_input_text}
    
    Rewritten input:{example_1_rewritten_input}

    Conversation history: {stringify_conversation_history(example_2_conv_history)}

    User input to rewrite: {example_2_input_text}

    Rewritten input: {example_2_rewritten_input}

    Conversation history: {stringify_conversation_history(example_4_conv_history)}

    User input to rewrite: {example_4_input_text}

    Rewritten input: {example_4_rewritten_input}
    
    TASK:

    Conversation history: {stringify_conversation_history(conversation_history)}
    
    User input to rewrite: {input_text}
    
    Rewritten input: '''

    # Set data for the request
    data = {'model': "gemma3:1b", 
            'prompt': rewrite_prompt}
    
    # Make request
    response = requests.post(url=generate_endpoint,
                                data=json.dumps(data))

    # Get response by iterating
    full_response = ""
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data = json.loads(line)
                if "response" in json_data:
                    full_response += json_data["response"]
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} - Line: {line}")
    
    # Return the rewritten input
    return full_response.strip(), rewrite_prompt

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
