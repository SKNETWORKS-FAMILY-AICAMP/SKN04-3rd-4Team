import streamlit as st
import openai
import time
import llm
from PIL import Image

st.set_page_config(page_title="Welcome BR", page_icon="🚅", layout="centered")

image = Image.open('https://github.com/user-attachments/assets/44e55662-1d8e-4159-8834-72b135461411')
st.image(image, width = 400)

st.title("🤖Welcome on board")
st.write("사수에게 물어보기 애매한 사항들을 질문해주세요.")

st.markdown("""
    <style>
        .stTextInput>div>div>input {
            font-size: 18px;
            padding: 10px;
            border-radius: 8px;
            border: 2px solid #4CAF50;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
        }
            
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            border: none;
            cursor: pointer;
        }

        .stButton>button:hover {
            background-color: #45a049;
        }

        .stWarning {
            background-color: #ff9800;
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
            
        .stSpinner {
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

question = st.text_input("질문을 입력하세요:", max_chars=100)

if st.button("🔍 답변 찾기", key="find_answer"):
    if question:
        with st.spinner("답변을 찾는 중입니다..."):
            try:
                
                response = llm.chain.invoke({
                    'question': question,
                    'language': '한국어'
                })

                st.markdown("#### 📝 답변")
                st.markdown(response)

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
    else:
        st.warning("💡 질문을 입력해 주세요.")

st.divider()

st.markdown("<br><br>", unsafe_allow_html=True)

st.subheader("자주 묻는 질문 (FAQ)")

faq_list = [
    {"question": "휴가 정책이 어떻게 되나요?", "answer": "신입사원은 첫 해에 연차 15일이 부여됩니다. 그 외의 정책은 내부 규정을 따릅니다."},
    {"question": "출퇴근 시간은 어떻게 되나요?", "answer": "일반적인 출퇴근 시간은 오전 9시부터 오후 6시까지입니다. 유연근무제가 제공됩니다."},
    {"question": "복리후생에 어떤 혜택이 포함되나요?", "answer": "복리후생에는 식사 지원, 건강검진, 경조사비, 복지 포인트 등이 포함됩니다."},
    {"question": "사내 교육 프로그램이 있나요?", "answer": "네, 신입사원 및 직무별 교육 프로그램을 제공하고 있으며, 연수 프로그램도 운영됩니다."},
    {"question": "승진 제도는 어떻게 운영되나요?", "answer": "승진은 연간 평가 결과와 회사 내부 규정을 바탕으로 결정됩니다."},
]

for i, faq in enumerate(faq_list):
    with st.expander(f"Q{i+1}: {faq['question']}"):
        st.write(f"A: {faq['answer']}")