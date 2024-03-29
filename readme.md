# Retrieval Augmented Generation (RAG) with LangChain and OpenAI
<a href="https://github.com/sienlonglim/LangChain"><img alt="Static Badge" src="https://img.shields.io/badge/github-black?style=flat-square&logo=github"></a> <img alt="GitHub commit activity (branch)" src="https://img.shields.io/github/commit-activity/t/sienlonglim/LangChain">
<a href="https://document-query-bot.streamlit.app/"><img alt="Static Badge" src="https://img.shields.io/badge/Streamlit%20App-red?style=flat-square&logo=streamlit&labelColor=white"></a> 

This project implements RAG using OpenAI's embedding models and LangChain's Python library.  The aim is to make a user-friendly RAG application with the ability to ingest data from multiple sources (word, pdf, txt, youtube, wikipedia)

Domain areas include:
- Document splitting
- Embeddings (OpenAI)
- Vector database (Chroma / FAISS)
- Semantic search types
- Retrieval chain

![Screenshot](images/framework.png)

## Upcoming works:
- Introduce conversation retriever and memory states
- Log embedding costs

### 20231216 Logging & token tracing:
- Added a logger
- Added callback to record cost of queries

### 20231215 Refactoring:
- rewrote main operations into OOP 
- added resource caching
- archived old code

### 20231213 MVP3:
- refactored code to use tempfile to utilize langchain's loaders
- added functionality to allow srt files
- added webbaseloader and youtube loader
- added an option to use Wikipedia as the retriever instead
- added brief documentation
- added debug mode (exceptions will be raised)

### 20231210 Refactoring and updates:
- refactored to work on modules
- allowed for wikipedia query with RAG

### 20231207 Refactoring and updates:
- refactored to use yaml config file
- allowed for txt and docx files

### 20231203 Updates:
- added status spinners
- updated tooltips

### 20231202 MVP2:
- Incorporated types of different query chains - restricted query, creative query
- Incorporated temperature settings
- Restructured functions to get functions
- Included explanations on the frontend and backend workings
- Included examples

### 20231201 Fixes and MVP1:
- chroma was changed to 0.3.29 for streamlit - did not work, reverted
- switched to FAISS vector db from Chroma db due to compatibility issues with Streamlit (sqlite versioning)
- removed pywin32 from library, streamlit is unable to install this dependency
