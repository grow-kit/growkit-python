#FastAPI 서버 실행부 (router 등록만)
from fastapi import FastAPI
from domains.evaluation.router import router as eval_router
from fastapi.middleware.cors import CORSMiddleware
from app.routes import upload
from app.routes import gpt_quiz

app = FastAPI()
# /analyze/evaluation 경로에 API 연결
app.include_router(eval_router) # prefix 날렸음

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api")
app.include_router(gpt_quiz.router, prefix="/api")
print("라우터 경로 목록:")
for route in app.routes:
    print(f"{route.path}  ⮕  {route.name}")