import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from translate import Translator

# Language dictionary mapping codes to full names
language_dict = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'nl': 'Dutch',
    'it': 'Italian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ar': 'Arabic'
}

# Function to perform text summarization
def summarize_text(input_text, api_key):
    # Initialize the Large Language Model with OpenAI
    model = OpenAI(temperature=0, openai_api_key=api_key)

    # Check if input text is empty
    if not input_text.strip():
        st.error("Please enter some text to summarize.")
        return None

    # Split the input text into smaller segments
    splitter = CharacterTextSplitter()
    text_segments = splitter.split_text(input_text)

    # Check if text segments are valid
    if not text_segments:
        st.error("Text splitting resulted in no valid segments.")
        return None

    # Convert text segments into Document objects
    document_list = [Document(page_content=segment) for segment in text_segments if segment.strip()]

    # Check if document list is valid
    if not document_list:
        st.error("No valid documents were created from the text segments.")
        return None

    # Set up the text summarization chain
    summarization_chain = load_summarize_chain(model, chain_type='map_reduce')

    # Run the summarization chain and return the result
    summarized_text = summarization_chain.run(document_list)
    return summarized_text

# Function to translate text
def translate_text(text, target_language_name):
    # Find the language code for the selected language
    target_language_code = [code for code, name in language_dict.items() if name == target_language_name][0]
    translator = Translator(to_lang=target_language_code)
    max_length = 500

    # Split text into chunks of max_length
    text_chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]

    # Translate each chunk and combine
    translated_chunks = [translator.translate(chunk) for chunk in text_chunks]
    translated_text = ' '.join(translated_chunks)

    return translated_text

# Streamlit App Configuration
st.set_page_config(page_title='Text Summarizer and Translator')
st.header('Text Summarizer and Translator Application')

# User input for text summarization
user_input = st.text_area("Paste the text you want to summarize and translate:", height=150)

# Dropdown for selecting translation language
language_choice = st.selectbox("Choose a language for translation:", list(language_dict.values()))

# Form for user submission
with st.form('text_process_form'):
    api_key = st.text_input('Enter your OpenAI API Key', type='password', help='Your API key should start with "sk-"')
    submit_button = st.form_submit_button('Process Text')

    # Process the form submission
    if submit_button and api_key.startswith('sk-'):
        with st.spinner('Processing...'):
            summarized_text = summarize_text(user_input, api_key)
            if summarized_text:
                translated_text = translate_text(summarized_text, language_choice)
                del api_key

# Display the results
if 'translated_text' in locals():
    st.subheader('Summarized Text:')
    st.write(summarized_text)
    st.subheader('Translated Text:')
    st.write(translated_text)