from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from domains.simulation.router import router as simulation_router

# 기존 라우터 - 평가
from domains.evaluation.router import router as eval_router

# 병합 대상 라우터들 - 교육
from domains.education.router import router as education_router

app = FastAPI()

# CORS 설정
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

# 라우터 등록
app.include_router(eval_router)  # /analyze/evaluation로 직접 라우팅하거나 내부 경로 설정
app.include_router(education_router, prefix="/api")
app.include_router(simulation_router, prefix="/simulation")  # 🔥 추가

# 라우터 확인용 출력 (선택사항)
print("라우터 경로 목록:")
for route in app.routes:
    print(f"{route.path}  ⮕  {route.name}")

# # main.py 맨 아래 추가
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)

