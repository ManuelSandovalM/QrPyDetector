from fastapi import FastAPI
from pydantic import BaseModel
from utils.qrDetector import QRDetector

from PIL import Image
import base64
from io import BytesIO

app = FastAPI()

qr_detector = QRDetector()

class RequestImage(BaseModel):
    imgBase64: str


@app.get("/")
def index():
    return {
        "status": 0,
        "description": "Service running"
    }


@app.post("/get-qrs")
def get_qrs_detection(data: RequestImage):
    detection_dict = {}
    
    try:
        input_image = Image.open(BytesIO(base64.b64decode(data.imgBase64))).convert('RGB')
    except Exception as e:
        return {
            'status': 2,
            'description': f"Error on input image, {e}",
            'response': {}
        }
    
    status, response = qr_detector.detect(input_image)

    for index, qr in enumerate(response):
        bbox = qr.getBoundingBox()
        detection_dict[str(index)] = [int(bbox.left), int(bbox.top), int(bbox.right), int(bbox.bottom)]

    if status == 0:

        return {
            'status': status,
            'description': "OK",
            'response': dict(detection_dict)
        }
    
    return {
        'status': status,
        'description': "Error on detection",
        'response': {}
    }

