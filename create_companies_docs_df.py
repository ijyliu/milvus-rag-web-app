
# %%
import pandas as pd
import os
import tqdm
# tqdm pandas
tqdm.tqdm.pandas()

# %%
# Load filenames in "Load Embeddings/Embeddings"
list_of_files = os.listdir("Load Embeddings/Embeddings")

# %%
# Dataframe
co_doc_df = pd.DataFrame({'filename': list_of_files})
co_doc_df

# %%
# Split on the underscore
co_doc_df['company'] = co_doc_df['filename'].str.split('_').str[0]
co_doc_df['doc_type'] = co_doc_df['filename'].str.split('_').str[1].str.split('.').str[0]
co_doc_df

# %%
# Seeing if Gemma can clean our doc_type column
import requests
import json

def get_doc_type_from_gemma(doc_type):
    """
    Function to get the doc_type from Gemma model.
    """
    data = {'model': "gemma3:1b", 
            'prompt': f'''
                      Please convert the following string representing a type of website terms of service document to a standard, title-cased format, making a reasonable interpretation as to missing or partial words when feasible. Return only the cleaned string.

                      Examples:

                      String: ElectronicRecordsDisclosu
                      Cleaned: Electronic Records Disclosure

                      String: WEBSITETERMSOFUSE
                      Cleaned: Website Terms of Use

                      String: PolicyrequirementsforGoo
                      Cleaned: Policy Requirements

                      String: CookieNotice_1
                      Cleaned: Cookie Notice 1

                      String: StackExchangeNetworkAcce
                      Cleaned: Stack Exchange Network Access Policy

                      String: LegalNotice
                      Cleaned: Legal Notice

                      String: {doc_type}
                      Cleaned: '''
            }
    
    chat_model_url = "http://host.docker.internal:3000"
    generate_endpoint = chat_model_url + '/api/generate'
    
    response = requests.post(url=generate_endpoint,
                             data=json.dumps(data))
    
    full_response = ""
    for line in response.iter_lines(decode_unicode=True):
        if line:
            try:
                json_data = json.loads(line)
                if "response" in json_data:
                    full_response += json_data["response"]
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e} - Line: {line}")

    to_return = full_response.strip()

    # print(f"Original doc_type: {doc_type}")
    # print(f"Cleaned doc_type: {to_return}")
    
    return to_return


# Create gemma_cleaned column, use tqdm to show progress
co_doc_df['gemma_cleaned'] = co_doc_df.progress_apply(lambda row: get_doc_type_from_gemma(row['doc_type']), axis=1)

# %%
# # Clean up Doc Type, 
# # Capitalize service
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('service', 'Service')
# # insert a space before each capital letter and then trim
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace(r'([a-z])([A-Z])', r'\1 \2', regex=True).str.strip()
# # Add a space after "Terms" if there is no space
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace(r'Terms(?! )', 'Terms ', regex=True)
# # Add a space after "Privacy" if there is no space
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace(r'Privacy(?! )', 'Privacy ', regex=True)
# # Replace "Disclosu" with "Disclosure"
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('Disclosu', 'Disclosure', case=False)
# # Capitalize everything after a space
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.title()
# # Capitalize policy
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('policy', 'Policy')
# # Capitalize notice
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('notice', 'Notice')
# # Capitalize everything after a space
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.title()
# # insert a space before each capital letter and then trim
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace(r'([a-z])([A-Z])', r'\1 \2', regex=True).str.strip()
# # replace copyrightpolicy with Copyright Policy, privacynotice with Privacy Notice, and terms of service with Terms of Service
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('Copyrightpolicy', 'Copyright Policy', case=False)
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('Privacynotice', 'Privacy Notice', case=False)
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('Termsofservice', 'Terms of Service', case=False)
# # conditionsof to Conditions Of
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('Conditionsof', 'Conditions Of', case=False)
# # replace Ofuse with Of Use
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('Ofuse', 'Of Use', case=False)
# # Po at end of string to Policy
# co_doc_df['doc_type'] = co_doc_df['doc_type'].str.replace('Po$', 'Policy', regex=True)
# co_doc_df

# %%
# Clean up dataframe
co_doc_df = co_doc_df[['company', 'doc_type', 'gemma_cleaned']].rename(columns={'company': 'Company', 'doc_type': 'Document Type'})

# %%
# Save to a csv file in "App/doc_df.csv"
co_doc_df.to_csv('App/doc_df.csv', index=False)
