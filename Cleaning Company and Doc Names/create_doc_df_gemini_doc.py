
# %%
# Packages
import pandas as pd
import os
import tqdm
# tqdm pandas
tqdm.tqdm.pandas()
from google import genai

# %%
# Configure the API key
with open('../Credentials/google_genai_api_key.txt', 'r') as file:
    google_genai_api_key = file.read().strip()
client = genai.Client(api_key=google_genai_api_key)

# %%
# Load filenames in "../Load Embeddings/Embeddings"
list_of_files = os.listdir("../Load Embeddings/Embeddings")

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
# Seeing if Gemini 2.0 Flash can clean our doc_type column

def get_doc_type_from_gemini(doc_type):
    """
    Function to get the doc_type from Gemini 2.0 Flash model.
    """
    try:

        prompt = f"""
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
            Cleaned:"""
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=prompt,
        )
        # Access the text from the candidate's content
        to_return = response.text.strip()
        
        return to_return
    except Exception as e:
        print(f"Error calling Gemini API for doc_type '{doc_type}': {e}")
        return doc_type # Return original doc_type in case of error

# Create gemini_cleaned column, use tqdm to show progress
co_doc_df['gemini_cleaned'] = co_doc_df.progress_apply(lambda row: get_doc_type_from_gemini(row['doc_type']), axis=1)

# %%
# Clean up dataframe
co_doc_df = co_doc_df[['company', 'doc_type', 'gemini_cleaned']].rename(columns={'company': 'Company', 'doc_type': 'Document Type'})

# %%
# Save to a csv file in "../App/doc_df.csv"
co_doc_df.to_csv('../App/doc_df.csv', index=False)
