from langchain_community.document_transformers import LongContextReorder

def reorder_documents(documents):
    # LongContextReorder 객체 생성: 긴 문맥을 재정렬하는 기능
    context_reorder = LongContextReorder()
    
    # 입력된 문서들을 재정렬
    documents_reordered = context_reorder.transform_documents(documents)
    
    # 재정렬된 문서의 내용을 줄바꿈으로 구분하여 하나의 문자열로 결합
    documents_joined = '\n'.join([document.page_content for document in documents_reordered])

    return documents_joined  # 재정렬된 문서의 내용을 반환