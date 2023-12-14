from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.retrievers import WikipediaRetriever
import streamlit as st

class VectorDB():
    def __init__(self, openai_api_key, config):
        self.api_key = openai_api_key
        self.config = config
        self.db_option = config['embedding_options']['db_option']

    def get_embedding_function(self):
        print('Creating embedding function')
        self.embedding_function = OpenAIEmbeddings(
            deployment="SL-document_embedder",
            model=self.config['embedding_options']['model'],
            show_progress_bar=True,
            openai_api_key = self.api_key) 
    
    def initialize_database(self, document_chunks, session_state):
        # Load it into FAISS
        print('Initializing vector_db')
        if self.db_option == 'FAISS':
            print('\tRunning in memory')
            self.vector_db = FAISS.from_documents(
                documents = document_chunks, 
                embedding = self.embedding_function)
        print('\tCompleted')

        # If successful, increment the usage based on number of documents
        if self.openai_api_key == session_state.openai_api_key_host:
            session_state.usage_counter += 1
        print(f'Current usage_counter: {session_state.usage_counter}')

    def get_llm(self, temperature : int):
        # Instantiate the llm object 
        print('Instantiating the llm')
        try:
            self.llm = ChatOpenAI(
                model_name=self.config['llm'],
                temperature=temperature,
                api_key=self.api_key)
        except Exception as e:
            print(e)
            st.error('Error occured, check that your API key is correct.', icon="🚨")
        else:
            print('\tCompleted')

    def get_chain(self, prompt_mode : str, source : str):
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
        if source == 'Uploaded documents / weblinks':
            qa_chain = RetrievalQA.from_chain_type(
                self.llm,
                retriever=self.vector_db.as_retriever(),
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
                self.llm,
                retriever=wiki_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_chain_prompt}
                )
            
        print('\tCompleted')
        return qa_chain


