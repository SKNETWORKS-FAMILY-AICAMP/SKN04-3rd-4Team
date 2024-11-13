import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import BM25Retriever
from utils import load_documents_from_folder, split_documents
import joblib
from dotenv import load_dotenv



load_dotenv()

folder_path = './data/신입사원'
documents = load_documents_from_folder(folder_path)
splitted_documents = split_documents(documents)

chroma_db = Chroma.from_documents(
    splitted_documents,
    embedding = OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name="example_collection",
    persist_directory='./save'
)



chroma_db = Chroma.from_documents(
    splitted_documents,
    embedding = OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name="example_collection",
    persist_directory='./save'
)

retriever = BM25Retriever.from_documents(
    splitted_documents,
    embedding = OpenAIEmbeddings(model='text-embedding-3-small'),
    collection_name="example_collection",
)

joblib.dump(retriever, './save/bm25_retriever_model.pkl')