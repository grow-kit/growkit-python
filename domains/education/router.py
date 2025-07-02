from fastapi import APIRouter, HTTPException, Body
from dotenv import load_dotenv
from openai import OpenAI
from fastapi import UploadFile, File
from domains.education.service import extract_text_from_pptx
import os

## gpt_quiz.py
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
router = APIRouter()


@router.post("/generate-quiz")
async def generate_gpt(text: str = Body(..., embed=True)):
    print("요청 들어옴:", text)
    if not text:
        raise HTTPException(status_code=400, detail="prompt가 없습니다.")

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "너는 교육 콘텐츠를 바탕으로 객관식 퀴즈를 생성하는 친절한 AI야. "
                                              "사용자가 제공한 텍스트를 기반으로 총 5개의 객관식 문제를 만들어줘. "
                                              "각 문제는 다음과 같은 구조로 JSON 배열로 출력해:\n\n"
                                              "- question: 질문 문자열\n"
                                              "- options: 보기 4개를 포함한 리스트\n"
                                              "- answer_index: 정답 보기의 인덱스 (0부터 시작)\n"
                                              "- explanation: 정답에 대한 간단한 해설 문자열\n\n"
                                              "형식은 다음 예시처럼 맞춰줘:\n"
                                              "[\n"
                                              "  {\n"
                                              "    \"question\": \"질문 내용\",\n"
                                              "    \"options\": [\"보기1\", \"보기2\", \"보기3\", \"보기4\"],\n"
                                              "    \"answer_index\": 2,\n"
                                              "    \"explanation\": \"정답에 대한 해설을 간단하게 설명해줘.\"\n"
                                              "  },\n"
                                              "  ... (총 5문제)\n"
                                              "]"
                 },
                {"role": "user", "content": text}
            ]
        )
        return {
            "prompt": text,
            "answer": response.choices[0].message.content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail="GPT 호출 실패: " + str(e))
# end def


## upload.py
UPLOAD_DIR = "input"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload-ppt")
async def upload_ppt(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    extracted_text = extract_text_from_pptx(file_path)
    return {"filename": file.filename, "text": extracted_text}
# end def