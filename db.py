from dotenv import load_dotenv

import os
import numpy as np
import pickle
import json
import bs4
from bs4 import BeautifulSoup
import requests
import uuid
import nest_asyncio
from tqdm import tqdm
from operator import itemgetter
from datetime import datetime
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from seaborn import load_dataset

from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from langchain_core.load import dumpd, dumps, load, loads
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_models import ChatOllama
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.prompts.few_shot import FewShotPromptTemplate
from langchain_core.example_selectors import (
    SemanticSimilarityExampleSelector,
    MaxMarginalRelevanceExampleSelector,
)
from langchain_core.output_parsers import (
    StrOutputParser,
    PydanticOutputParser,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
)
from langchain.output_parsers.structured import (
    ResponseSchema,
    StructuredOutputParser,
)
from langchain.output_parsers.pandas_dataframe import PandasDataFrameOutputParser
from langchain.output_parsers.datetime import DatetimeOutputParser
from langchain.output_parsers.fix import OutputFixingParser
from langchain.output_parsers.retry import RetryOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    ConfigurableField,
)

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    PyPDFium2Loader,
    PDFMinerLoader,
    PyPDFDirectoryLoader,
    PDFPlumberLoader,
    UnstructuredExcelLoader,
    DataFrameLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    WebBaseLoader,
    TextLoader,
    DirectoryLoader,
    JSONLoader,
    ArxivLoader,
)

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    RecursiveJsonSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.storage import (
    LocalFileStore,
    InMemoryByteStore,
    InMemoryStore,
)
from langchain.embeddings import CacheBackedEmbeddings

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_chroma import Chroma

from langchain_teddynote.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import (
    ContextualCompressionRetriever,
    BM25Retriever,
    EnsembleRetriever,
    ParentDocumentRetriever,
    TimeWeightedVectorStoreRetriever,
)
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever


folder_path = './data/신입사원'
documents = []

# 폴더 내 모든 .docx 파일을 로드
for filename in os.listdir(folder_path):
    if filename.endswith('.docx'):
        file_path = os.path.join(folder_path, filename)
        loader = UnstructuredWordDocumentLoader(file_path)
        documents.extend(loader.load())
