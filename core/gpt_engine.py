import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
from fastapi import HTTPException

## 최신버전은 이 방법을 사용해야 한다 ##
load_dotenv()  # .env 파일에서 환경변수 로드
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 명시적으로 전달


# 메뉴얼 받아오기 함수
def fetch_manual(manual_id: int) -> str:
    url = f"http://localhost:9000/api/manuals/{manual_id}"
    res = requests.get(url)

    ## 메뉴얼이 업을 경우
    if res.status_code != 200:
        raise ValueError(f"메뉴얼 ID {manual_id}에 해당하는 데이터를 찾을 수 없습니다.")

    return res.json().get("content", "")
# end def


# 문항 생성 함수
def generate_question_with_manual(manual_id: int):
    # 메뉴얼이 DB에 등록되어있지 않을 경우 예외 처리
    try:
        manual = fetch_manual(manual_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # 메뉴얼 내용이 비어있을 경우 예외 처리
    if not manual.strip():
        raise HTTPException(status_code=400, detail="매뉴얼 내용이 비어 있어 문항을 생성할 수 없습니다.")

    print("메뉴얼 내용:", manual)

    prompt = f"""
    너는 지금 서비스직 평가를 위한 시뮬레이션 문제를 하나 생성하는 역할이야.
    스타벅스 매장에서 실제 발생할 수 있는 단 하나의 상황을 설정해 줘.

    [교육 매뉴얼]
    {manual}

    [조건]
    - 메뉴얼을 참고하여, 서비스직원이 실제 겪을 수 있는 상황 한 가지를 작성해.
    - 절대 예시, 번호, 유형 등을 넣지 마.
    - 무조건 하나의 상황만 작성해.
    - 출력은 단 한 문장만. 다른 말, 설명, 앞말 없이 바로 "상황 설명 : ..." 형식으로만 출력.

    [출력 예시]
    상황 설명 : 한 고객이 주문한 음료를 받고 나서, 맛이 다르다며 불만을 제기하고 있습니다.

    ※ 절대 여러 상황을 나열하지 말 것.
    ※ 절대 문제 번호, 예시, 유형 등의 문구를 포함하지 말 것.
    ※ 반드시 아래 형식처럼 시작할 것: 상황 설명 :
    """

    print("GPT 요청 프롬프트:", prompt)

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            # {"role": "system",
            #  "content": "절대 과거 대화, 형식, 문장을 기억하지 마. 지금 이 요청만 완전히 처음 보는 것처럼 처리해. 이전 대화와 관련된 추론은 금지야. 이건 완전히 새로운 대화야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content
# end def


# 분석 및 피드백 함수
def generate_feedback(question: str, answer: str, emotion: dict):
    prompt = f"""
질문: {question}
답변: {answer}
감정 분석: {emotion}

위 내용을 바탕으로 피드백과 5점 만점 기준 점수를 채점해 주세요.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    content = response.choices[0].message.content
    return {
        "feedback": content,
        "score": 4.5
    }
# end def
