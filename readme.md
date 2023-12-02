# Retrieval Augmented Generation (RAG) with LangChain and OpenAI
This project implements RAG using OpenAI's embedding models and LangChain's Python library

20231201 Fixes:
- chroma was changed to 0.3.29 for streamlit - did not work, reverted
- switched to FAISS vector db from Chroma db due to compatibility issues with Streamlit (sqlite versioning)
- removed pywin32 from library, streamlit is unable to install this dependency

Upcoming works:
- Include explanations on the frontend and backend workings
- Include examples
- Incorporate types of different query chains - strict query, imaginative query, general query
- Restructured functions or maybe use OOP instead
- Add logger