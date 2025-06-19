
# %%
# Packages
import pandas as pd
import os
import tqdm
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

# %%
# Split on the underscore
co_doc_df['company'] = co_doc_df['filename'].str.split('_', n = 1).str[0]
co_doc_df['doc_type'] = co_doc_df['filename'].str.split('_', n = 1).str[1].str.split('.').str[0]

def get_cleaned_company_and_doc_type_from_gemini(company_name, doc_type):
    """
    Function to get cleaned company name and document type from Gemini 2.5 Flash model
    in a single API call, providing full context.
    Returns a tuple (cleaned_company, cleaned_doc_type).
    """
    try:
        prompt = f'''
        Please clean up the following company name and document type strings.

        For the company name, make a reasonable interpretation as to how to fix things by resolving issues such as whether things belong to separate words, and what should go in the place of any missing or corrupted characters.

        For the document type, convert it to a standard, title-cased format, making a reasonable interpretation as to missing or partial words when feasible.

        Return your response as first the clean company name, then a pipe (|) character, then the clean document type.

        Examples:

        Company: NestlÃ©
        Doc Type: TermsandConditions
        Cleaned: Nestlé|Terms and Conditions

        Company: Zoosk
        Doc Type: ElectronicRecordsDisclosu
        Cleaned: Zoosk|Electronic Records Disclosure

        Company: CrowdStrike
        Doc Type: WEBSITETERMSOFUSE
        Cleaned: CrowdStrike|Website Terms of Use

        Company: StackOverflow
        Doc Type: StackExchangeNetworkAcce
        Cleaned: Stack Overflow|Stack Exchange Network Access Policy

        Company: GoogleAnalytics
        Doc Type: PolicyrequirementsforGoo
        Cleaned: Google Analytics|Policy Requirements for Google Analytics

        Company: Honeywell
        Doc Type: CookieNotice_1
        Cleaned: Honeywell|Cookie Notice 1

        Company: Punkt
        Doc Type: LegalNotice
        Cleaned: Punkt|Legal Notice

        Company: EVEOnline
        Doc Type: EVEOnlineEndUserLicen
        Cleaned: EVE Online|EVE Online End User License Agreement

        Company: UniversitÃ CommercialeLuigiBocconi
        Doc Type: CookiePolicy
        Cleaned: Università Commerciale Luigi Bocconi|Cookie Policy

        Company: {company_name}
        Doc Type: {doc_type}
        Cleaned:'''
        
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite-preview-06-17',
            contents=prompt,
        )
        
        # Access the text
        response_text = response.text.strip()
        # Split the response into cleaned company and doc type
        cleaned_data = response_text.split('|')
        # Return original values if length is not 2
        if len(cleaned_data) != 2:
            print(f"Unexpected response format for company '{company_name}' and doc_type '{doc_type}': {response_text}")
            return company_name, doc_type
        # Otherwise return cleaned values
        cleaned_company, cleaned_doc_type = cleaned_data[0].strip(), cleaned_data[1].strip()
        return cleaned_company, cleaned_doc_type
    except Exception as e:
        print(f"Error calling Gemini API for company '{company_name}' and doc_type '{doc_type}': {e}")
        return company_name, doc_type # Return original values in case of error

# %%
# Apply the single cleaning function to the DataFrame
# Use tqdm.pandas() for progress tracking
co_doc_df[['cleaned_company', 'cleaned_doc_type']] = co_doc_df.progress_apply(
    lambda row: get_cleaned_company_and_doc_type_from_gemini(row['company'], row['doc_type']),
    axis=1,
    result_type='expand' # This expands the tuple return into two new columns
)

# %%
# Set column order - filename, company, cleaned_company, doc_type, cleaned_doc_type
co_doc_df = co_doc_df[['filename', 'company', 'cleaned_company', 'doc_type', 'cleaned_doc_type']]

# %%
# Save to an Excel file in the working directory
output_filename = 'doc_df_gemini.xlsx'
co_doc_df.to_excel(output_filename, index=False)

# %%
print(f"Cleaned data saved to {os.path.abspath(output_filename)}")
