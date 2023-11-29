import streamlit as st

def load_documents(documents):
  chunk_size=1000
  chunk_overlap=50
  

def generate_response(user_input):
    
    # Instantiate the llm object 
    

    # Build prompt template
    

    # Build QuestionAnswer chain

    # Query and Response
    st.info('Query Response:', icon='📕')
    st.info("result")
    st.write(' ')
    st.info('Sources', icon='📚')


def main():
  # Main page area
  st.markdown("### :house: Welcome to Sien Long's LangChain RAG Project")
  st.info("Current loaded document:\nSubscribed: Why the Subscription Model Will Be Your Company's Future - and What to Do About It", icon='ℹ️')
  st.write('Enter your API key on the sidebar to begin')

  # Sidebar
  openai_api_key =st.sidebar.text_input("Enter your API key")
  documents = st.file_uploader(label = 'Upload documents to for embedding to VectorDB', 
                                help = 'Current acceptable files (pdf, txt)',
                                type = ['pdf', 'txt'], 
                                accept_multiple_files=True)
  if st.sidebar.button('Upload', type='primary'):
     with st.sidebar.spinner('Uploading...'):
        load_documents(documents)

  # Query form and response
  with st.form('my_form'):
    user_input = st.text_area('Enter prompt:', 'What is the book about and who is the author?')

    if not openai_api_key.startswith('sk-'):
      st.warning('Please enter your OpenAI API key!', icon='⚠')

    if st.form_submit_button('Submit') and openai_api_key.startswith('sk-'):
      with st.spinner('Loading...'):
        generate_response(user_input)

# Main
if __name__ == '__main__':
   main()
