import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
from PyPDF2 import PdfReader
import re

def load_split_clean(file_input : list, use_splitter = True, 
                     remove_leftover_delimiters = True,
                     remove_pages = False,
                     chunk_size=1000, chunk_overlap=50,
                     front_pages_to_remove : int = None, last_pages_to_remove : int = None, 
                     delimiters_to_remove : list = ['\t', '\n', '   ', '  ']):
    
    # Main list of all LangChain Documents
    documents = []

    # Handle file by file
    for file_index, file in enumerate(file_input):
        reader = PdfReader(file)
        print(f'Original pages of document: {len(reader.pages)}')
        for key, value in reader.metadata.items():
            if 'title' in key:
                pdf_title = value
            else:
                pdf_title = 'uploaded_file_' + str(file_index)
                
        # Remove pages
        if remove_pages:
            for _ in range(front_pages_to_remove):
                del reader.pages[0]
            for _ in range(last_pages_to_remove):
                reader.pages.pop()
            print(f'Number of pages after skipping: {len(reader.pages)}')

        # Each file will be split into LangChain Documents and kept in a list
        document = []

        # Loop through each page and extract the text and write in metadata
        if use_splitter:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                separators = ['\n\n', ' '])
            for page_number, page in enumerate(reader.pages):
                page_chunks = splitter.split_text(page.extract_text()) # splits into a list of strings (chunks)

                # Write into Document format with metadata
                for chunk in page_chunks:
                    document.append(Document(page_content=chunk, 
                                             metadata={'source': pdf_title , 'page':page_number+1}))
            
        else:
            # This will split purely by page
            for page_number, page in enumerate(reader.pages):
                document.append(Document(page_content=page.extract_text(), 
                                      metadata={'source': pdf_title , 'page':page_number+1}))
                i += 1
        

        # Remove all left over the delimiters and extra spaces
        if remove_leftover_delimiters:
            for chunk in document:
                for delimiter in delimiters_to_remove:
                    chunk.page_content = re.sub(delimiter, ' ', chunk.page_content)

        documents.extend(document)

    print(f'Number of document chunks extracted: {len(documents)}')
    return documents


def embed_to_vector_db(documents):
  pass

def initialize_models():
  # Instantiate the llm object 
  model_name = 'gpt-3.5-turbo-1106'
  llm = ChatOpenAI(model_name=model_name, temperature=0, api_key=openai_api_key)

  
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
      llm,
      retriever=vector_db.as_retriever(),
      return_source_documents=True,
      chain_type_kwargs={"prompt": qa_chain_prompt}
  )

def generate_response(user_input):
  # Query and Response
  result = qa_chain({"query": user_input})
  st.info('Query Response:', icon='📕')
  st.info(result["result"])
  st.write(' ')
  st.info('Sources', icon='📚')
  for document in result['source_documents']:
      st.write(document.page_content + '\t- ' + document.metadata['source'] + '(pg ' + document.metadata['page'] + ')')
      st.write('')


def main():
  '''
  Main Function
  '''
  st.set_page_config(page_title="LangChain RAG Project ", page_icon='🏠')
  docs = None

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
          vector_db = embed_to_vector_db()

  # Main page area
  st.markdown("### :rocket: Welcome to Sien Long's Document Query Bot")
  st.info(f"Current loaded document(s) \n\n {docs}", icon='ℹ️')
  st.write('Enter your API key on the sidebar to begin')

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
