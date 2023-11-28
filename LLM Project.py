import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.markdown("### :house: Welcome to Sien Long's LangChain RAG Project")
st.markdown("#### Current loaded document:\nSubscribed: Why the Subscription Model Will Be Your Company's Future - and What to Do About It")
st.write('Enter your API key on the sidebar to begin')
openai_api_key =st.sidebar.text_input("Enter your API key")

def generate_response(user_input):
    
    # Instantiate the llm object 
    llm_name = 'gpt-3.5-turbo-1106'
    llm = ChatOpenAI(model_name=llm_name, temperature=0, api_key=openai_api_key)

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

    result = qa_chain({"query": user_input})
    st.info('Query Response:', icon='📕')
    st.info(result["result"])
    st.write(' ')
    st.info('Sources', icon='📚')
    for document in result['source_documents']:
        st.write(document.page_content)
        st.write('')


with st.form('my_form'):
  user_input = st.text_area('Enter text:', 'What is the book about?')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(user_input)
