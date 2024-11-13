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

folder_path = './data/신입사원'
documents = []

# 폴더 내 모든 .docx 파일을 로드
for filename in os.listdir(folder_path)[:2]:
    if filename.endswith('.docx'):
        file_path = os.path.join(folder_path, filename)
        loader = UnstructuredWordDocumentLoader(file_path)
        documents.extend(loader.load())

chroma_db = Chroma.from_documents(
    documents,
    embedding = OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name="example_collection",
    persist_directory='./save'
)

