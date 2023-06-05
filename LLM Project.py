import streamlit as st
from langchain.llms import OpenAI

st.markdown("### :house: Welcome to Sien Long's LLM Project with LangChain")
st.write('Enter your API key on the sidebar to begin')
openai_api_key =st.sidebar.text_input("Enter your API key")

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_api_key)
    st.info(llm(input_text))

with st.form('my_form'):
  user_input = st.text_area('Enter text:', 'How many things that I tell you can you remember?')
  submitted = st.form_submit_button('Submit')
  if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and openai_api_key.startswith('sk-'):
    generate_response(user_input)
