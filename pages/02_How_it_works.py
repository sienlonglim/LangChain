import streamlit as st
st.set_page_config(page_title="How it works")

st.markdown('# :rocket: How it works')

st.write("This Streamlit app is built following the RAG framework with the use of LangChain's library. \n\
         RAG, standing for Retrieval Augmented Generation is a framework that allows us to retrieve facts from an external knowledge base. \
         This allows our LLM to respond based on the context of these facts, reducing the occurences of hallucinations")
st.image('images/framework.png')
st.markdown("<p style='text-align: center'>The Retrieval Augmented Generation (RAG) framework</p>", unsafe_allow_html=True)

st.write('The main idea in RAG is implemented through the use of vectors embeddings. These embeddings are vectorised chunks of information which are created \
         through the use of an LLM model and stored in vector databases, allowing us to search for the most relevant piece of information via \
         cosine similarity (Semantic search). Let me break down the process part by part')


st.header('Part 1 : Document splitting')
st.image('images/p1.png')
st.write("A document is simple programming terms is usually one long string. The first step is to split these texts into smaller chunks while maintaining their meaning. \
         This will allow us to embed the information effective for storage. This application uses LangChain's recursiveSplitter to reduce each document into smaller chunks, \
         while capturing the metadata on title and page for the information.")


st.header('Part 2 : Embedding')
st.image('images/p2.png')
st.write("In this next step, we use OpenAI's embedding model - 'text-embedding-ada-002' to create embeddings for each chunk. \
         An embedding is a vector (list) of floating point numbers. \
         The purpose is to later find the distance between two vectors to measure their relatedness (semantic search). \
         These embeddings are stored in a vector database, I started with using Chroma database, but due to versioning issues with Streamlit, \
         the latest version of this app runs FAISS. The vector database is stored in memory for each run, it is possible to persist the database as well.")


st.header('Part 3 : Semantic search and Prompt chaining')
st.write("This is where RAG comes in. We create a chain - a pipeline of how to process our query. \
         When a prompt is given, the embedding model will convert this prompt to an embedding (like how we did in part 2) \
         and perform a semantic search. The results retrieved would be the top N embeddings which are closest to prompt, \
         which will be converted back to their original strings")
st.image('images/p3.png')
st.write("Now that we have the most relevant information, we feed these into a customise prompt (prompt engineering), adding context to the original prompt. \
         The customize prompt is then used to query the LLM for a response. This app also pulls the semantic search result and display it on-screen, \
         giving us visibility on what was actually used to produce that query")