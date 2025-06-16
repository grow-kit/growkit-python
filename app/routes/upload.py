from fastapi import APIRouter, UploadFile, File
from app.services.ppt_parser import extract_text_from_pptx
import os

router = APIRouter()
UPLOAD_DIR = "input"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload-ppt")
async def upload_ppt(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    extracted_text = extract_text_from_pptx(file_path)
    return {"filename": file.filename, "text": extracted_text}