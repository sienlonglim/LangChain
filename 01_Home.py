import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import re

def load_split_clean(documents):
  chunk_size=1000
  chunk_overlap=50
  

def generate_response(user_input):
    
    # Instantiate the llm object 
    model_name = 'gpt-3.5-turbo-1106'
    model = ChatOpenAI(model_name=model_name, temperature=0, api_key=openai_api_key)

    
    persist_directory = 'ignore/chroma/'
    embedding_function = OpenAIEmbeddings(api_key=openai_api_key)
    vector_db = Chroma(embedding_function = embedding_function,
                      persist_directory=persist_directory)

    # Build prompt template
    template = """Use the following pieces of context to answer the question at the end. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer. \
    Use three sentences maximum. Keep the answer as concise as possible. 
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    qa_chain_prompt = PromptTemplate.from_template(template)

    # Build QuestionAnswer chain
    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_db.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_chain_prompt}
    )

    # Query and Response
    result = qa_chain({"query": user_input})
    st.info('Query Response:', icon='📕')
    st.info(result["result"])
    st.write(' ')
    st.info('Sources', icon='📚')
    for document in result['source_documents']:
        st.write(document.page_content)
        st.write('')


def main():
  st.set_page_config(page_title="LangChain RAG Project ", page_icon='🏠')
  docs = None
  # docs = ["Subscribed: Why the Subscription Model Will Be Your Company's Future - and What to Do About It"]

  # Main page area
  st.markdown("### :rocket: Welcome to Sien Long's Document Query Bot")
  st.info(f"Current loaded document(s) \n\n {docs}", icon='ℹ️')
  st.write('Enter your API key on the sidebar to begin')

  # Sidebar
  with st.sidebar:
    openai_api_key =st.text_input("Enter your API key")
    documents = st.file_uploader(label = 'Upload documents for embedding to VectorDB', 
                                  help = 'Current acceptable files (pdf, txt)',
                                  type = ['pdf', 'txt'], 
                                  accept_multiple_files=True)
    if st.button('Upload', type='primary'):
      with st.spinner('Uploading...'):
          docs = load_split_clean(documents)

  # Query form and response
  with st.form('my_form'):
    user_input = st.text_area('Enter prompt:', 'What is the book about and who is the author?')

    if not openai_api_key.startswith('sk-'):
      st.warning('Please enter your OpenAI API key!', icon='⚠')

    if st.form_submit_button('Submit', type='primary') and openai_api_key.startswith('sk-'):
      with st.spinner('Loading...'):
        generate_response(user_input)

# Main
if __name__ == '__main__':
   main()
