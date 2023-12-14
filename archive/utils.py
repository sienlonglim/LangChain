import re
import pysrt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, YoutubeLoader, WebBaseLoader, TextLoader
from langchain.schema import Document
from tempfile import NamedTemporaryFile

def remove_delimiters(document_chunks, config : dict):
    '''
    Helper function to remove remaining delimiters in document chunks
    '''
    delimiters_to_remove = config['splitter_options']['delimiters_to_remove']
    for chunk in document_chunks:
        for delimiter in delimiters_to_remove:
            chunk.page_content = re.sub(delimiter, ' ', chunk.page_content)
    return document_chunks

def remove_chunks(document_chunks : list, config : dict):
    '''
    Helper function to remove any unwanted document chunks after splitting
    '''
    front = config['splitter_options']['front_chunk_to_remove']
    end = config['splitter_options']['last_chunks_to_remove']
    # Remove pages
    for _ in range(front):
        del document_chunks[0]
    for _ in range(end):
        document_chunks.pop()
        print(f'\tNumber of pages after skipping: {len(document_chunks)}')
    return document_chunks

def get_pdf(temp_file_path : str, title : str, config : dict, splitter : object):
    '''
    Function to process PDF files
    '''
    loader = PyMuPDFLoader(temp_file_path) #This loader preserves more metadata

    if splitter:
        document_chunks = splitter.split_documents(loader.load())
    else:
        document_chunks = loader.load()

    if config['splitter_options']['remove_chunks']:
        document_chunks = remove_chunks(document_chunks, config)

    if 'title' in document_chunks[0].metadata.keys():
        title = document_chunks[0].metadata['title']

    print(f"\t\tOriginal no. of pages: {document_chunks[0].metadata['total_pages']}")
    print(f'\t\tExtracted no. of chunks: {len(document_chunks)}')

    return title, document_chunks

def get_txt(temp_file_path : str, title : str, config : dict, splitter : object):
    '''
    Function to process TXT files
    '''
    loader = TextLoader(temp_file_path, autodetect_encoding=True)

    if splitter:
        document_chunks = splitter.split_documents(loader.load())
    else:
        document_chunks = loader.load()

    # Update the metadata
    for chunk in document_chunks:
        chunk.metadata['source'] = title
        chunk.metadata['page'] = 'N/A'

    if config['splitter_options']['remove_chunks']:
        document_chunks = remove_chunks(document_chunks, config)

    print(f'\t\tExtracted no. of chunks: {len(document_chunks)}')
    return title, document_chunks

def get_srt(temp_file_path : str, title : str, config : dict, splitter : object):
    '''
    Function to process SRT files
    '''
    subs = pysrt.open(temp_file_path)

    text = ''
    for sub in subs:
        text += sub.text
    document_chunks = [Document(page_content=text)]
    
    if splitter:
        document_chunks = splitter.split_documents(document_chunks)

    print(document_chunks)
    # Update the metadata
    for chunk in document_chunks:
        chunk.metadata['source'] = title
        chunk.metadata['page'] = 'N/A'

    if config['splitter_options']['remove_chunks']:
        document_chunks = remove_chunks(document_chunks, config)

    print(f'\t\tExtracted no. of chunks: {len(document_chunks)}')
    return title, document_chunks

def get_docx(temp_file_path : str, title : str, config : dict, splitter : object):
    '''
    Function to process DOCX files
    '''
    loader = Docx2txtLoader(temp_file_path)

    if splitter:
        document_chunks = splitter.split_documents(loader.load())
    else:
        document_chunks = loader.load()

    # Update the metadata
    for chunk in document_chunks:
        chunk.metadata['source'] = title
        chunk.metadata['page'] = 'N/A'

    if config['splitter_options']['remove_chunks']:
        document_chunks = remove_chunks(document_chunks, config)

    print(f'\t\tExtracted no. of chunks: {len(document_chunks)}')
    return title, document_chunks

def get_youtube_transcript(url : str, config : dict, splitter : object):
    '''
    Function to retrieve youtube transcript and process text
    '''
    loader = YoutubeLoader.from_youtube_url(
        url, 
        add_video_info=True,
        language=["en"],
        translation="en"
    )

    if splitter:
        document_chunks = splitter.split_documents(loader.load())
    else:
        document_chunks = loader.load_and_split()

    if config['splitter_options']['remove_chunks']:
        document_chunks = remove_chunks(document_chunks, config)
    
    title = document_chunks[0].metadata['title']

    print(f'\t\tExtracted no. of chunks: {len(document_chunks)}')
    return title, document_chunks

def get_html(url : str, config : dict, splitter : object):
    '''
    Function to process websites via HTML files
    '''
    loader = WebBaseLoader(url)

    if splitter:
        document_chunks = splitter.split_documents(loader.load())
    else:
        document_chunks = loader.load_and_split()

    if config['splitter_options']['remove_chunks']:
        document_chunks = remove_chunks(document_chunks, config)
    
    title = document_chunks[0].metadata['title']

    print(f'\t\tExtracted no. of chunks: {len(document_chunks)}')
    return title, document_chunks

def get_chunks(uploaded_files : list, weblinks : list, config : dict):
    '''
    Main function to split or load all the documents into chunks
    '''
    splitter = None

    remove_leftover_delimiters = config['splitter_options']['remove_leftover_delimiters']
    use_splitter = config['splitter_options']['use_splitter']
    split_by_token = config['splitter_options']['split_by_token']

    if use_splitter:
        if split_by_token:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=config['splitter_options']['chunk_size'],
                chunk_overlap=config['splitter_options']['chunk_overlap'],
                separators = config['splitter_options']['chunk_separators']
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=config['splitter_options']['chunk_size'],
                chunk_overlap=config['splitter_options']['chunk_overlap'],
                separators = config['splitter_options']['chunk_separators']
                )
        
    # Main list of all LangChain Document Chunks
    document_chunks_full = []
    document_names = []
    print(f'Splitting documents: total of {len(uploaded_files)}')

    # Handle file by file
    for file_index, file in enumerate(uploaded_files):

        # Get the file type and file name
        file_type = file.name.split('.')[-1]
        print(f'\tSplitting file {file_index+1} : {file.name}')
        file_name = ''.join(file.name.split('.')[:-1])

        with NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(file.read())

        # Handle different file types
        if file_type =='pdf':
            title, document_chunks = get_pdf(temp_file_path, file_name, config, splitter)
        elif file_type == 'txt':
            title, document_chunks = get_txt(temp_file_path, file_name, config, splitter)
        elif file_type == 'docx':
            title, document_chunks = get_docx(temp_file_path, file_name, config, splitter)
        elif file_type == 'srt':
            title, document_chunks = get_srt(temp_file_path, file_name, config, splitter)

        # Remove all left over the delimiters and extra spaces
        if remove_leftover_delimiters:
            document_chunks = remove_delimiters(document_chunks, config)

        document_names.append(title)
        document_chunks_full.extend(document_chunks)

    # Handle youtube links:
    if weblinks[0] != '':
        print(f'Splitting weblinks: total of {len(weblinks)}')
        
        # Handle link by link
        for link_index, link in enumerate(weblinks):
            print(f'\tSplitting link {link_index+1} : {link}')
            if 'youtube' in link:
                title, document_chunks = get_youtube_transcript(link, config, splitter)
            else:
                title, document_chunks = get_html(link, config, splitter)

            document_names.append(title) 
            document_chunks_full.extend(document_chunks)

    print(f'\tNumber of document chunks extracted in total: {len(document_chunks_full)}')
    return document_chunks_full, document_names
