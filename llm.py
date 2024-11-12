
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

import os
import pickle
import json
import bs4
import uuid
import nest_asyncio
from operator import itemgetter
from datetime import datetime
from pydantic import BaseModel, Field
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
from utils import reorder_documents
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
import numpy as np
import db


load_dotenv()
set_llm_cache(InMemoryCache())

# BM25Retriever를 사용하여 문서에서 검색을 수행하는 리트리버 생성
retriever_b25 = BM25Retriever.from_documents(db.documents)
retriever_b25.k = 1  # 반환할 결과의 개수를 1로 설정

# Chroma를 사용하여 문서로부터 임베딩 데이터베이스 생성
db_chroma = Chroma.from_documents(
    db.documents,  # 사용될 문서
    embedding=OpenAIEmbeddings(model='text-embedding-3-small'),  # 사용할 임베딩 모델 지정
)

# Chroma 데이터베이스를 리트리버로 변환하고 검색할 결과의 개수를 1로 설정
retriever_chroma = db_chroma.as_retriever(search_kwargs={'k': 1})

# 앙상블 리트리버 생성: 두 개의 리트리버를 결합하여 결과를 통합
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_b25, retriever_chroma],  # 사용할 리트리버 목록
    weights=[0.5, 0.5],  # 각 리트리버에 대한 가중치 (합이 1이 되도록 설정)
).configurable_fields(
    # 앙상블 리트리버의 가중치를 설정할 수 있는 필드 추가
    weights=ConfigurableField(
        id='ensemble_weights',  # 필드의 고유 ID
        name='Ensemble Weights',  # 사용자에게 보여질 필드 이름
        description='앙상블된 두 retriever의 비율 (비율의 합은 1)'  # 필드에 대한 설명
    )
)



# 프롬프트 템플릿 정의: reference, question, language를 포함
template = '''
너는 신입사원을 대상으로 기업에 알려주는 봇이다.
회사 내부 규정과 정책에 대한 정보를 바탕으로 아래의 질문에 상세히 답하라. 
답변한 내용 마지막에는 참고 서식으로 reference 내용도 추가해서 답변하여라 
#### 제공된 정보(Reference)
{reference}

#### 질문:
{question}

#### 주의사항:
질문에 대해 정확하고 구체적인 답변을 제공해야 한다.
회사 내부 규정에 대한 답변은 공식적인 언어를 사용하라.
답변은 주어진 언어로 작성하라: {language}
회사 내부 문서에 없는 내용을 질문할 경우 "해당 내용은 관련 부서에 문의하시기바랍니다."로 출력하라

#### 예시 질문:
"연차 휴가는 어떻게 신청하나요?"
"급여 지급일은 언제인가요?"
"퇴사 절차에 대해 설명해 주세요."
'''

# 템플릿으로부터 프롬프트 생성
prompt = PromptTemplate.from_template(template)

# OpenAI의 Chat 모델 초기화 (gpt-4o-mini 모델 사용)
model = ChatOpenAI(model_name='gpt-4o-mini')

# 출력 파서를 초기화 (문자열 출력을 처리)
parser = StrOutputParser()

# 체인 구성: 데이터 흐름을 정의
chain = (
    {
        # 'reference' 키에 대해 여러 처리를 정의
        'reference': itemgetter('question')  # 질문에서 reference를 추출
        | retriever_chroma  # Chroma 리트리버로부터 데이터를 검색
        | RunnableLambda(reorder_documents),  # 문서를 재정렬
        'question': itemgetter('question'),  # 질문을 그대로 가져오기
        'language': itemgetter('language'),  # 언어 정보를 가져오기
    }
    | prompt  # 프롬프트 템플릿에 데이터 결합
    | model  # 모델에 프롬프트 전달하여 응답 생성
    | parser  # 모델의 응답을 파싱
)

# 체인을 사용하여 질문을 실행하고 응답을 받음
response = chain.invoke({
    'question': '10년차 과장의 휴가는 몇개야',  # 질문: 삼성전자의 최근 이슈
    'language': '한국어'  # 언어: 한국어로 응답 요청
})

# 응답 출력
print(response)