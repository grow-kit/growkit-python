import openai

openai.api_key = "your-key"

def generate_feedback(question: str, answer: str, emotion: dict):
    prompt = f"""
    질문: {question}
    답변: {answer}
    감정 분석: {emotion}

    위 내용을 바탕으로 피드백과 5점 만점 기준 점수를 채점해 주세요.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    content = response.choices[0].message.content
    return {
        "feedback": content,  # 여기선 파싱 로직 생략
        "score": 4.5          # 샘플 점수
    }