#FastAPI 서버 실행부 (router 등록만)
from fastapi import FastAPI
from domains.evaluation.router import router as eval_router

app = FastAPI()
# /analyze/evaluation 경로에 API 연결
app.include_router(eval_router) # prefix 날렸음
