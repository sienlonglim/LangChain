from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import WikipediaRetriever
import streamlit as st

def increment_api_usage_count(session_state : int, increment : int):
    session_state.usage_counter += increment

def get_embeddings(openai_api_key : str, documents : list,  config : dict, session_state : dict):
    # create the open-source embedding function
    model = config['embedding_options']['model']
    db_option = config['embedding_options']['db_option']
    # persist_directory = config['embedding_options']['persist_directory']

    embedding_function = OpenAIEmbeddings(deployment="SL-document_embedder",
                                        model=model,
                                        show_progress_bar=True,
                                        openai_api_key = openai_api_key) 

    # Load it into FAISS
    print('Initializing vector_db')
    if db_option == 'FAISS':
        print('\tRunning in memory')
        vector_db = FAISS.from_documents(documents = documents, 
                                        embedding = embedding_function)
    print('\tCompleted')

    # If successful, increment the usage based on number of documents
    if openai_api_key == session_state.openai_api_key_host:
        docs_processed = len(session_state.doc_names)
        increment_api_usage_count(session_state.usage_counter, docs_processed)
        print(f'Current usage_counter: {session_state.usage_counter}')
    return vector_db

def get_llm(openai_api_key : str, temperature : int, config : dict):
    model_name = config['llm']
    # Instantiate the llm object 
    print('Instantiating the llm')
    try:
        llm = ChatOpenAI(model_name=model_name, 
                        temperature=temperature, 
                        api_key=openai_api_key)
    except Exception as e:
        print(e)
        st.error('Error occured, check that your API key is correct.', icon="🚨")
    else:
        print('\tCompleted')
    return llm 

def get_chain(session_state : dict, prompt_mode : str, source : str):
    print('Creating chain from template')
    if prompt_mode == 'Restricted':
        # Build prompt template
        template = """Use the following pieces of context to answer the question at the end. \
        If you don't know the answer, just say that you don't know, don't try to make up an answer. \
        Keep the answer as concise as possible. 
        Context: {context}
        Question: {question}
        Helpful Answer:"""
        qa_chain_prompt = PromptTemplate.from_template(template)
    elif prompt_mode == 'Creative':
        # Build prompt template
        template = """Use the following pieces of context to answer the question at the end. \
        If you don't know the answer, you may make inferences, but make it clear in your answer. 
        Context: {context}
        Question: {question}
        Helpful Answer:"""
        qa_chain_prompt = PromptTemplate.from_template(template)

    # Build QuestionAnswer chain
    if source == 'Uploaded documents':
        qa_chain = RetrievalQA.from_chain_type(
            session_state.llm,
            retriever=session_state.vector_db.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt}
            )
    elif source == 'Wikipedia':
        wiki_retriever = WikipediaRetriever(
            top_k_results = 5,
            lang= "en", 
            load_all_available_meta = False,
            doc_content_chars_max = 4000,
            features="lxml"
            )
        qa_chain = RetrievalQA.from_chain_type(
            session_state.llm,
            retriever=wiki_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt}
            )
        
    print('\tCompleted')
    return qa_chain

def get_response(user_input : str, qa_chain : object):
    # Query and Response
    print('Getting response from server')
    result = qa_chain({"query": user_input})
    return result
