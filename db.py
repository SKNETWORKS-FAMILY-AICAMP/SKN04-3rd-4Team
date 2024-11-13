import os
from langchain_community.document_loaders import (
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from utils import load_documents_from_folder, split_documents

folder_path = './data/신입사원'
documents = load_documents_from_folder(folder_path)
splitted_documents = split_documents(documents)

chroma_db = Chroma.from_documents(
    splitted_documents,
    embedding = OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name="example_collection",
    persist_directory='./save'
)