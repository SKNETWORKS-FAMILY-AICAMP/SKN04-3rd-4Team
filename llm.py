from dotenv import load_dotenv
from utils import reorder_documents
from operator import itemgetter
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
import numpy as np
import template_llm

load_dotenv()
# set_llm_cache(InMemoryCache())

# BM25Retriever를 사용하여 문서에서 검색을 수행하는 리트리버 생성

# Chroma를 사용하여 문서로부터 임베딩 데이터베이스 생성
db_chroma = Chroma(
    collection_name="example_collection",
    embedding_function = OpenAIEmbeddings(model='text-embedding-3-small'),
    persist_directory= './save'
)

# Chroma 데이터베이스를 리트리버로 변환하고 검색할 결과의 개수를 1로 설정
retriever_chroma = db_chroma.as_retriever(search_kwargs={'k': 1})




'''
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
# 프롬프트 템플릿 정의: reference, question, language를 포함
template = template_llm.template

# 템플릿으로부터 프롬프트 생성
prompt = PromptTemplate.from_template(template)

# OpenAI의 Chat 모델 초기화 (gpt-4o-mini 모델 사용)
model = ChatOpenAI(model_name='gpt-4o-mini')

# 출력 파서를 초기화 (문자열 출력을 처리)
parser = StrOutputParser()

# 체인 구성: 데이터 흐름을 정의x
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


# # 체인을 사용하여 질문을 실행하고 응답을 받음
# response_ = chain.invoke({
#     'question': streamlit.question,  # 질문: 삼성전자의 최근 이슈
#     'language': '한국어'  # 언어: 한국어로 응답 요청
# })
