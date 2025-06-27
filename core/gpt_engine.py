import os
from openai import OpenAI
from dotenv import load_dotenv
import requests
from fastapi import HTTPException
import httpx
import re

## 최신버전은 이 방법을 사용해야 한다 ##
load_dotenv()  # .env 파일에서 환경변수 로드
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # 명시적으로 전달


# 메뉴얼 받아오기 함수
async def fetch_manual(manual_id: int) -> str:
    url = f"http://localhost:9000/api/manuals/{manual_id}"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)

    if res.status_code != 200:
        raise ValueError(f"메뉴얼 ID {manual_id}에 해당하는 데이터를 찾을 수 없습니다.")

    data = res.json()
    content = data.get("content")

    if not content:
        raise ValueError(f"메뉴얼 ID {manual_id}에 content가 존재하지 않습니다.")

    return content


# end def


# 문항 생성 함수
async def generate_question_with_manual(manual_id: int):
    # 메뉴얼이 DB에 등록되어있지 않을 경우 예외 처리
    try:
        manual = await fetch_manual(manual_id)
    except ValueError as e:
        print("❌ 메뉴얼 없음 예외 발생:", e)
        raise HTTPException(status_code=404, detail=str(e))

    # 메뉴얼 내용이 비어있을 경우 예외 처리
    if not manual.strip():
        print("⚠️ 메뉴얼 내용이 공백이거나 없음")
        raise HTTPException(status_code=400, detail="매뉴얼 내용이 비어 있어 문항을 생성할 수 없습니다.")

    print("메뉴얼 내용:", manual)

    prompt = f"""
    너는 지금 서비스직 평가를 위한 시뮬레이션 문제를 하나 생성하는 역할이야.
    스타벅스 매장에서 실제 발생할 수 있는 단 하나의 상황을 설정해 줘.

    [교육 매뉴얼]
    {manual}

    [조건]
    - 메뉴얼을 참고하여, 서비스직원이 실제 겪을 수 있는 상황 한 가지를 작성해. 
    - 너무 매뉴얼에 갇히지 말고 창의성을 발휘해서 너가 생각하는 카페에서 일어날 수 있는 다양한 상황 중에 하나를 골라서 제시해. (예시: 진상 손님, 나이가 많은 노인, 메뉴 추천 요청하는 손님, 무리한 요청을 하는 손님 등)
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
        temperature=0.7
    )
    return response.choices[0].message.content


# end def


# 분석 및 피드백 함수
# def generate_feedback(question: str, answer: str, emotion: dict):
#     prompt = f"""
# 질문: {question}
# 답변: {answer}
#
# 위 내용을 바탕으로 피드백과 5점 만점 기준 점수를 채점해 주세요.
# """
#
#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7
#     )
#
#     content = response.choices[0].message.content
#     return {
#         "feedback": content,
#         "score": 4.5
#     }
# # end def

def extract_scores_from_text(feedback_text: str) -> dict:
    """GPT 출력에서 항목별 점수를 파싱"""
    criteria = ["친절도", "문제해결능력", "소통능력", "전문성", "감정조절", "태도"]
    scores = {}

    for criterion in criteria:
        # 예: "친절도: 4.5" 또는 "친절도 : 5"
        match = re.search(rf"{criterion}\s*[:：]\s*(\d+(\.\d+)?)", feedback_text)
        if match:
            scores[criterion] = round(float(match.group(1)))
        else:
            scores[criterion] = 0  # 점수 인식 실패 시 기본 0점

    return scores
# end def

# 메뉴얼 받아오기 함수
async def fetch_manual(manual_id: int) -> str:
    url = f"http://localhost:9000/api/manuals/{manual_id}"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)

    if res.status_code != 200:
        raise ValueError(f"메뉴얼 ID {manual_id}에 해당하는 데이터를 찾을 수 없습니다.")

    data = res.json()
    content = data.get("content")

    if not content:
        raise ValueError(f"메뉴얼 ID {manual_id}에 content가 존재하지 않습니다.")

    return content
# end def

# 평가 기준 받아오기 함수
async def fetch_criteria(criteria_id: int) -> str:
    url = f"http://localhost:9000/api/criteria/{criteria_id}"
    async with httpx.AsyncClient() as client:
        res = await client.get(url)
    if res.status_code != 200:
        raise ValueError("평가 기준 없음")
    return res.json().get("guideline", "")


def generate_feedback_with_criteria(question, answer, emotion, manual, criteria):
    # 기본 반환 형식
    result = {
        "question": question,
        "answer": answer.strip(),  # 항상 포함!
        "feedback": "",
        "score": {}
    }

    # 1. 답변 무효 조건 처리
    if answer.strip() == "음성 인식 실패" or len(answer.strip()) < 5:
        result["feedback"] = "⚠️ 답변이 정상적으로 인식되지 않아 평가가 불가능합니다. 문의 후 재평가 요청을 진행해 주세요."
        result["score"] = {}
        return result

    # 2. 정상 분석 수행
    gaze = emotion.get("gaze", "알 수 없음")
    head = emotion.get("head", "알 수 없음")

    additional_notes = ""
    if gaze == "알 수 없음":
        additional_notes += "- 시선 정보가 없으므로 '소통능력'과 '태도' 평가에는 반영하지 마세요.\n"
    if head == "알 수 없음":
        additional_notes += "- 고개 움직임 정보가 없으므로 '감정조절'과 '전문성' 평가에는 반영하지 마세요.\n"

    prompt = f"""
[메뉴얼]
{manual}

[평가 기준]
{criteria}

[문제]
{question}

[답변]
{answer}

[시선/고개 움직임 분석 결과]
- 시선 방향: {gaze}
- 고개 움직임: {head}

[주의 사항]
- 정면을 잘 응시했다면 '소통능력'과 '태도' 항목의 점수를 높게 주세요.
- 고개 움직임이 안정적이라면 '감정조절'과 '전문성' 점수도 높게 평가해 주세요.
- 반대로 시선을 회피하거나, 고개를 자주 움직이면 해당 항목 점수를 낮춰 주세요.
{additional_notes}

위 내용을 참고하여, 다음 6가지 항목에 대해 각각 5점 만점 기준으로 채점하고, 간단한 설명과 함께 총평도 작성해 주세요:

1. 친절도
2. 문제해결능력
3. 소통능력
4. 전문성
5. 감정조절
6. 태도

[출력 예시]
- 친절도: 4.5
- 문제해결능력: 4.0
...
- 총평: 전반적으로 안정적인 태도와 정중한 언행을 보였습니다.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    feedback_text = response.choices[0].message.content
    result["feedback"] = feedback_text
    result["score"] = extract_scores_from_text(feedback_text)

    return result