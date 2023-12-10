import streamlit as st
import yaml
from utils import *
from vector_db import *

def get_settings():
    '''
    Handles initializing of session_state variables
    '''
    # Create empty list of document names
    if 'doc_names' not in st.session_state:
        st.session_state.doc_names = None
    
    # Load config if not yet loaded
    if 'config' not in st.session_state:
        with open('config.yml', 'r') as file:
            st.session_state.config = yaml.safe_load(file)

    # Create an API call counter, to cap usage
    if 'usage_counter' not in st.session_state:
        st.session_state.usage_counter = 0

    # Check if host api key is enabled
    if 'openai_api_key_host' not in st.session_state:
        if st.session_state.config['enable_host_api_key']:
            st.session_state.openai_api_key_host = st.secrets['openai_api_key']
        else:
            st.session_state.openai_api_key_host = 'NA'

def main():
    '''
    Main Function
    '''
    # Load configs and check for session_states
    st.set_page_config(page_title="Document Query Bot ")
    result = None
    source = None
    get_settings()

    # Sidebar
    with st.sidebar:
        # API option
        if st.session_state.usage_counter >= 5:
            api_option = st.radio('API key usage', ['Host API key usage cap reached!'])
        else:
            api_option = st.radio('API key usage', ['Use my own API key', 'Use host API key (capped)'], help='Cap is counted by number of documents uploaded (max 5 in total)')
        if (api_option == 'Use my own API key') or (api_option == 'Host API key usage cap reached!'):
            openai_api_key =st.text_input("Enter your API key")
        else:
            openai_api_key = st.session_state.openai_api_key_host
        
        # Document uploader
        documents = st.file_uploader(label = 'Upload documents for embedding to VectorDB', 
                                    help = 'Overwrites any existing files uploaded',
                                    type = ['pdf', 'txt', 'docx'], 
                                    accept_multiple_files=True)
        if st.button('Upload', type='primary') and documents:
            with st.status('Uploading... (this may take a while)', expanded=True) as status:
                try:
                    st.write("Splitting documents...")
                    document_chunks_full, st.session_state.doc_names = get_chunks(documents, st.session_state.config)
                    st.write("Creating embeddings...")
                    st.session_state.vector_db = get_embeddings(openai_api_key, document_chunks_full, st.session_state.config, st.session_state)
                except Exception as e:
                    print(e)
                    status.update(label='Error occured.', state='error', expanded=False)
                else:
                    status.update(label='Embedding complete!', state='complete', expanded=False)

    # Main page area
    st.markdown("### :rocket: Welcome to Sien Long's Document Query Bot")

    # Info bar
    if st.session_state.doc_names:
        doc_name_display = ''
        for doc_count, doc_name in enumerate(st.session_state.doc_names):
            doc_name_display += str(doc_count+1) + '. ' + doc_name + '\n\n'
    else:
        doc_name_display = 'No documents uploaded yet!'
    st.info(f"Current loaded document(s): \n\n {doc_name_display}", icon='ℹ️')
    if not openai_api_key:
        st.write('Enter your API key on the sidebar to begin')

    # Query form and response
    with st.form('my_form'):
        user_input = st.text_area('Enter prompt:', value='Enter prompt here')
        source = st.selectbox('Query from:', ('Uploaded documents', 'Wikipedia'),
                              help="Selecting Wikipedia will instead prompt from Wikipedia's repository")

        # Select for model and prompt template settings
        prompt_mode = st.selectbox('Choose mode of prompt', ('Restricted', 'Creative'), 
                                help='Restricted mode will reduce chances of LLM answering using out of context knowledge')
        temperature = st.select_slider('Select temperature', options=[x / 10 for x in range(0, 21)], 
                                    help='0 is recommended for restricted mode, 1 for creative mode. \n\
                                        Going above 1 creates more unpredictable responses and takes longer.')
        
        # Display error if no API key given
        if not openai_api_key.startswith('sk-'):
            if openai_api_key == 'NA':
                st.warning('Host key currently not available, please use your own OpenAI API key!', icon='⚠')
            else:
                st.warning('Please enter your OpenAI API key!', icon='⚠')

        # Submit a prompt
        if st.form_submit_button('Submit', type='primary') and openai_api_key.startswith('sk-'):
            with st.spinner('Loading...'):
                try:
                    st.session_state.llm = get_llm(openai_api_key, temperature, st.session_state.config)
                    st.session_state.qa_chain = get_chain(st.session_state, prompt_mode, source)
                    result = get_response(user_input, st.session_state.qa_chain)
                except Exception as e:
                    print(e)
                    st.error('Error occured, unable to process response!', icon="🚨")

            if result:
                # Display the result
                st.info('Query Response:', icon='📕')
                st.info(result["result"])
                st.write(' ')
                st.info('Sources', icon='📚')
                for document in result['source_documents']:
                    st.write(document.page_content + '\n\n' + document.metadata['source'] + ' (pg ' + document.metadata.get('page', 'NA') + ')')
                    st.write('-----------------------------------')
                print('\tCompleted')

# Main
if __name__ == '__main__':
   main()
