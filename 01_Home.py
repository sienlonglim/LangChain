import streamlit as st
import yaml
from modules.InfoLoader import InfoLoader
from modules.VectorDB import VectorDB
import logging
import os

@st.cache_resource
def configure_logging(file_path=None, streaming=None, level=logging.INFO):
    '''
    Initiates the logger, runs once due to caching
    '''
    # streamlit_root_logger = logging.getLogger(st.__name__)

    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add a filehandler to output to a file
    if file_path:
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Add a streamhandler to output to console
    if streaming:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)
    

    return logger

def initialize_session_state():
    '''
    Handles initializing of session_state variables
    '''
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
        
@st.cache_resource
def get_resources():
    '''
    Initializes the customer modules
    '''
    return InfoLoader(st.session_state.config), VectorDB(st.session_state.config)

def main():
    '''
    Main Function for streamlit interface
    '''
    # Load configs and check for session_states
    st.set_page_config(page_title="Document Query Bot ")
    initialize_session_state()    
    logger = configure_logging()
    loader, vector_db = get_resources()  

    #------------------------------------ SIDEBAR ----------------------------------------#
    with st.sidebar:
        # API option, whether to use host's API key (must be enabled by config), and also to cap usage
        if st.session_state.usage_counter >= 5:
            api_option = st.radio(
                'API key usage', 
                ['Host API key usage cap reached!'], 
                help='Only a maximum of 5 artefacts are allowed, further uploading has been disabled')
        else:
            api_option = st.radio(
                'API key usage', 
                ['Use my own API key', 'Use host API key (capped)'], 
                help='Cap is counted by number of documents uploaded (max 5 in total)'
                )
        # Response to API option
        if (api_option == 'Use my own API key'):
            openai_api_key =st.text_input("Enter your API key")
        else:
            openai_api_key = st.session_state.openai_api_key_host
        
        # Document uploader
        uploaded_files = st.file_uploader(
            label = 'Upload documents for embedding', 
            help = 'Overwrites any existing files uploaded',
            type = ['pdf', 'txt', 'docx', 'srt'], 
            accept_multiple_files=True
            )
        st.write('And / Or')
        weblinks = st.text_area(label = 'Retrieve from website or youtube video transcript (Enter every link on a new line)').split('\n')

        if api_option != 'Host API key usage cap reached!':
            if st.button('Upload', type='primary') and (uploaded_files or weblinks):
                with st.status('Uploading... (this may take a while)', expanded=True) as status:
                    try:
                        st.write("Splitting documents...")
                        loader.get_chunks(uploaded_files, weblinks)
                        st.write("Creating embeddings...")
                        vector_db.create_embedding_function(openai_api_key)
                        vector_db.initialize_database(loader.document_chunks_full, loader.document_names)

                    except Exception as e:
                        logger.error('Exception during Splitting / embedding', exc_info=True)
                        status.update(label='Error occured.', state='error', expanded=False)
                    else:
                        # If successful, increment the usage based on number of documents
                        if openai_api_key == st.session_state.openai_api_key_host:
                            st.session_state.usage_counter += len(loader.document_names)
                            logger.info(f'Current usage counter: {st.session_state.usage_counter}')
                        logger.info(f'Uploaded: {loader.document_names}')
                        status.update(label='Embedding complete!', state='complete', expanded=False)

    #------------------------------------- MAIN PAGE -----------------------------------------#
    st.markdown("## :rocket: Welcome to Sien Long's RAG Bot")

    # Info bar
    if vector_db.document_names:
        doc_name_display = ''
        for doc_count, doc_name in enumerate(vector_db.document_names):
            doc_name_display += str(doc_count+1) + '. ' + doc_name + '\n\n'
    else:
        doc_name_display = 'No documents uploaded yet!'
    st.info(f"Current loaded document(s): \n\n {doc_name_display}", icon='ℹ️')
    if (not openai_api_key.startswith('sk-')) or (openai_api_key=='NA'):
        st.write('Enter your API key on the sidebar to begin')

    # Query form and response
    with st.form('my_form'):
        user_input = st.text_area('Enter prompt:', value='Enter prompt here')
        source = st.selectbox('Query from:', ('Uploaded documents / weblinks', 'Wikipedia'),
                              help = "Selecting Wikipedia will instead prompt from Wikipedia's repository"
                              )

        # Select for model and prompt template settings
        prompt_mode = st.selectbox(
            'Choose mode of prompt', 
            options = ('Restricted', 'Unrestricted'),
            help='Restricted mode will reduce chances of LLM answering using out of context knowledge',
            # disabled = missing_api_key
            )
        temperature = st.select_slider(
            'Select temperature', 
            options=[x / 10 for x in range(0, 21)],
            help='0 is recommended for restricted mode, 1 for creative mode. \n\
                Going above 1 creates more unpredictable responses and takes longer.',
            # disabled = missing_api_key
            )
        
        # Display error if no API key given
        if not openai_api_key.startswith('sk-'):
            if openai_api_key == 'NA':
                st.warning('Host key currently not available, please use your own OpenAI API key!', icon='⚠')
            else:
                st.warning('Please enter your OpenAI API key!', icon='⚠')

        #----------------------------------------- Submit a prompt ----------------------------------#
        if st.form_submit_button('Submit', type='primary') and openai_api_key.startswith('sk-'):
            with st.spinner('Loading...'):
                try:
                    result = None
                    vector_db.create_llm(
                        openai_api_key,
                        temperature
                    )
                    vector_db.create_chain(
                        prompt_mode,
                        source
                    )
                    result = vector_db.get_response(user_input)
                except Exception as e:
                    logger.error('Exception during Querying', exc_info=True)
                    st.error('Error occured, unable to process response!', icon="🚨")

            if result:
                # Display the result
                st.info('Query Response:', icon='📕')
                st.info(result["result"])
                st.write(' ')
                st.info('Sources', icon='📚')
                for document in result['source_documents']:
                    st.write(document.page_content + '\n\n' + document.metadata['source'] + ' (pg ' + document.metadata.get('page', 'na') + ')')
                    st.write('-----------------------------------')

if __name__ == '__main__':
   main()
