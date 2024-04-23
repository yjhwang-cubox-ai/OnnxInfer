from fastapi import FastAPI, File, UploadFile
from onnx_infer import TextRecognition
import os
import cv2
import time

app = FastAPI()
@app.post('/photo')
async def upload_photo(file: UploadFile):
    recog = TextRecognition(model_path="models/svtr-base_20e_st_mj_vn_20240404.onnx", ops_path="", font_path="arial.ttf", font_size=12, dict_path="dicts/vietnamese_unicode.txt")
    
    UPLOAD_DIR = "./photo"  # 이미지를 저장할 서버 경로
    
    content = await file.read()
    filename = f"{file.filename}.jpg"  # uuid로 유니크한 파일명으로 변경
    with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
        fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)
        
    img = cv2.imread(os.path.join(UPLOAD_DIR, filename))
    txt = recog.recognition(img)
    
    return txt