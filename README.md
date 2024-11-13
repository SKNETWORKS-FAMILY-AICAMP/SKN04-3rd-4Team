# 🚄BRT🚄
<p align="center"><img src="https://github.com/user-attachments/assets/19fa9782-5046-4bb4-b383-6cf727140fb4" width="1000" height="300"/></p>
<hr>

### 🚆 팀명 : 어중이와 떠중이
 
### 🚅 팀원

<p align="center">
	<img src="https://github.com/user-attachments/assets/1cd0e421-e3f6-45fa-856c-e0ae349a20be" width="163" height="163"/>
	<img src="https://github.com/user-attachments/assets/7a392f78-9d9b-4b89-bbe8-e914160f2951" width="163" height="163"/>
	<img src="https://github.com/user-attachments/assets/856a13c7-89e9-4b30-83ad-5560912c5ac5" width="163" height="163"/>
	<img src="https://github.com/user-attachments/assets/4bcf5798-7ca6-4ea3-b89b-073536e13fe2" width="163" height="163"/>
	<img src="https://github.com/user-attachments/assets/66761246-36f9-4261-a45c-02dcafd07e30" width="163" height="163"/>
	<img src="https://github.com/user-attachments/assets/d19a1b1e-3e1d-48a7-b3a4-99fe3d9ae584" width="163" height="163"/>
</p>


|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;안준용&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;박진효&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;권오셈&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;전욱진&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;하상집&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;김현재&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|
|--------------------------|-------------------------|--------------------------|-------------------------|-------------------------|-------------------------|


### 조원 역할

안준용 : 데이터 전처리, 백터 DB 구축, 랭체인 구축  
박진효 : 벡터DB 구축, 랭체인 구축  
권오셈 : 랭체인 구축, streamlit UI 개발, 깃관리  
전욱진 : 데이터 수집, 데이터 전처리, 로고 디자인  
하상집 : 랭체인 구축, streamlit UI 개발, 문서화 및 ReadME작성  
김현재 : streamlit UI 개발, 문서화 및 ReadME 작성  

<hr>

### 🚆 프로젝트 개요
가상의 회사 신입사원을 위한  기업 내 정보 검색 챗봇을 구현하여 회사 내부 규정과 정책에 대한 정보를 신입사원이 쉽게 접근하고 활용할 수 있도록 합니다.

<hr>

### 🚅 프로젝트 목표
신입사원 Q&A 챗봇 개발 : 회사 규정, 가이드라인, 정책 등을 기반으로 신입사원들이 자주 묻는 질문에 대해 답변해주는 Q&A 챗봇 개발하여 이를 통해 신입사원들이 복잡한 규정을 일일이 확인하지 않고 빠르게 답변을 얻을 수 있게하고, 궁금증을 해결하여 신속하게 회사에 적응하도록 돕는 것이 목표입니다.

정보 제공 신뢰성 : AI 챗봇의 일반적인 문제인 할루시네이션(잘못된 정보 생성)을 최소화하여, 내부 공식 문서 기반 신뢰할 수 있는 답변을 제공하는 것이 목표입니다.



<hr>

### 🔨 기술 스택
<div>
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/streamlit%20-%23FF0000.svg?style=for-the-badge&logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/openai-0769AD?style=for-the-badge&logo=openai&logoColor=black">
<img src="https://img.shields.io/badge/langchain-F7DF1E?style=for-the-badge&logo=langchain&logoColor=black">
<img src="https://img.shields.io/badge/git-F05032?style=for-the-badge&logo=git&logoColor=white">
<img src="https://github.com/user-attachments/assets/c8cd01e7-6ce6-46db-8cc3-b13286829cf3" width="163" height="27"/>
</div>


<hr>

### Prerequisites

```cmd
pip install -r requirements.txt
```

<hr>

### Usage

```cmd
streamlit run main.py
```

<hr> 

### Data
기존 철도 회사의 규정, 가이드라인, 정책 문서 등으로  데이터를 수집하고 신입사원에게 필요한 문서를 선택적으로 수집하고 정리하였습니다. 데이터는 주로 다음과 같은 내용이 포함되어 있습니다.

각 카테고리별 규정 문서를 신입사원이 챗봇을 통해 해당 정보를 질문하면 직접 조회 및 답변할 수 있도록 구성하였습니다.

1) 인사 및 근무 규정
2) 복지 및 혜택
3) 행동 및 윤리
4) 교육 및 훈련
5) 보수 및 포상
6) 안전 및 관리 규정
7) 자산 및 물품 관리
8) 정보 및 보안
9) 사무 및 내부 통제
10) 벤처 및 지식 재산

<hr>


### Preprocess

데이터 전처리 과정은 다음과 같은 단계로 진행되었습니다.

텍스트 정제 : 수집한 규정 문서에 불필요한 문구를 편집하여 정리합니다.


임베딩 벡터화 : 각 문서를 OpenAIEmbedddings모델을 적용하여 텍스트 데이터를 벡터화하고, Chroma를 사용하여 임베딩 데이터 베이스를 생성 후 저정합니다.

RAG 구축 : Chroma DB를 리트리버로 변환하여 적접한 답변을 제공할수 있도록 프롬프트 템플릿, Model(chatopenAI-gpt-4o-mini), parser를 체인을 구성하여 최적의 답변을 제공합니다.


<hr>

### System Architecture

<img src="https://github.com/user-attachments/assets/be78a2ba-2d72-461f-95e7-0b95ec4a8fd1">
<hr>

### 수행 결과
Q . "교육 훈련 규정문서에서 제1조에 대해 말해줘"

<table>
  <tr>
    <td style="width: 50%; vertical-align: top;">
      <h3>할루시네이션 최소화 전 </h3>
      <p>챗봇이 정책 및 규정을 정확히 참고 못하고 엉뚱한 응답을 제공함.</p>
      <ul>
        <li>A1.  제1(목적)는 운전교육센터 운영규정의 목적을 규정하고 있습니다. 이 조항의 내용은 철도안전 법 제16조 제3항에 따라, 주식회사 비알(주)BR)이 시행하는 고속철도차량 운전면허 교육 및 직무 교육의 내용, 방법, 운전업무종사자의 안전교육 및 직무교육의 내용, 방법, 절차, 평가 등에 관한 필요한 사항을 정하는 것을 목적으로 하고 있습니다. 이는 고속철도 차량 운전업무 안정성과 효율성을 높이고, 철도 운영에 필요한 인재를 양성하기 위한 기준을 마련하는 데 그 의의가 있습니다. 이 조항은 규정의 전반적인 방향성과 목표를 제시하며, 교육을 통해 확보하고자 하는 안전과 전문성을 강조하고 있습니다.</li>
      </ul>
    </td>
    <td style="width: 50%; vertical-align: top;">
      <h3>할루시네이션 최소화 후</h3>
      <p>챗봇이 특정 질문에 정확한 문서에 답을 제공하는 모습을 제공함</p>
      <ul>
        <li>A2. 제1조 교육훈련 규정에 따르면, 본 규정의 목적은 직원의 직무능력 향상과 전문성을 강화하기 위한 세계적인 교육 훈련을 제공하는 것입니다. 이를 통해 직원들이 지속적으로 성장하고 조직의 목표에 기여할 수 있도록 지원합니다. 또한, 교육 훈련의 대상, 방법, 진행 절차 및 평가 기준에 대한 구체적인 사항이 포함되어 있으며, 직원들은 정기적으로 제공되는 교육에 참여해야할 의무가 있습니다. 회사는 교육 훈련의 기회를 균등하게 제공하며, 모든 직원이 공정하게 혜택을 받을 수 있도록 노력하고 있습니다.</li>
        <li>참고 서식: 교육 훈련 규정 제1조 내용 요약.</li>
      </ul>
    </td>
  </tr>
</table>



<hr>

### 한 줄 회고

***회고 작성***  
안준용 : 어쩌다보니 팀장을 하게 되었는데 준비도 안된 팀장 따라오느라 고생하셨습니다.  
박진효 : 프로젝트 시작 전부터 걱정이 많았는데 생각보다 복잡하지 않고 큰 문제 없이 마무리 지어서 다행입니다. 고생하신 팀원들에게 고맙다는 말을 전하고 싶습니다.  
권오셈 : 훌륭한 분들이랑 팀플하게 되어서 배울거 많은 좋은 시간이었습니다. 고생하셨습니다.  
전욱진 : 시간이 다소 촉박한 감이 있었는데 의욕적으로 임해주셔서 크게 문제 없이 끝나게 도와주신 팀원분들께 감사드립니다.   
하상집 : 실제 rag 구축하는 과정에서 어려움이 있었지만, 각자 맡은 바 잘 하셔서 의미 있는 시간이었습니다. 고생하셨습니다.  
김현재 : 옆에서 서포트하면서 많이 배웠습니다. 다들 고생하셨습니다.  