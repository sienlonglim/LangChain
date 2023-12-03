# Retrieval Augmented Generation (RAG) with LangChain and OpenAI
This project implements RAG using OpenAI's embedding models and LangChain's Python library

20231201 Fixes and MVP1:
- chroma was changed to 0.3.29 for streamlit - did not work, reverted
- switched to FAISS vector db from Chroma db due to compatibility issues with Streamlit (sqlite versioning)
- removed pywin32 from library, streamlit is unable to install this dependency

20231202 MVP2:
- Incorporated types of different query chains - restricted query, creative query
- Incorporated temperature settings
- Restructured functions to get functions
- Included explanations on the frontend and backend workings
- Included examples

20231203 updates:
- added status spinners
- updated tooltips

Upcoming works:
- Add logger
- Allow for txt and docx files