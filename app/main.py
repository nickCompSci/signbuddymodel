import base64
import os
import uuid

import numpy as np
import cv2
from dotenv import load_dotenv
from jose import jwt
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI, HTTPException, status,  Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from .models import ImageResponseData, ImageRequestData
from .security import fetchSecrets
import importlib.resources

app = FastAPI()
load_dotenv()

MODEL_AUTH0_AUDIENCE= fetchSecrets("MODEL_AUTH0_AUDIENCE")
MODEL_AUTH0_ISSUER= fetchSecrets("MODEL_AUTH0_ISSUER")
MODEL_AUTH0_KEY= fetchSecrets("MODEL_AUTH0_KEY")
MODEL_AUTH0_ALOGIRTHMS= fetchSecrets("MODEL_AUTH0_ALOGIRTHMS")
ALLOWED_ORIGINS = fetchSecrets("ALLOWED_ORIGINS")
ALLOWED_HOSTS = fetchSecrets("ALLOWED_HOSTS")


ALPHABET_MAPPING_DICT = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4,
    'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
    'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19,
    'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25
}

origins = [
    fetchSecrets("ALLOWED_ORIGINS")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

module_path = os.path.dirname(os.path.abspath(__file__))
yolo_model_path = os.path.join(module_path, 'best.pt')
saved_image_dir = os.path.join(module_path, 'images')
model = YOLO(yolo_model_path) 

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@app.middleware("http")
async def ValidateJwt(request: Request, call_next):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={'WWW-Authenticate': "Bearer"},
    )
    authorization_header = request.headers.get("Authorization")

    if not authorization_header:
        raise credentials_exception
    bearer_token = authorization_header.split("Bearer ")[1]
    try:
        jwt.decode(token=bearer_token, audience=MODEL_AUTH0_AUDIENCE, issuer=MODEL_AUTH0_ISSUER, key=MODEL_AUTH0_KEY, algorithms=[MODEL_AUTH0_ALOGIRTHMS])
        response = await call_next(request)
        return response
    except Exception:
        raise credentials_exception
   

@app.post("/inference", response_model=ImageResponseData)
async def process_image(image_data: ImageRequestData):
  letter_to_detect = [ALPHABET_MAPPING_DICT[image_data.letter]]
  # remove details about image from string 
  base64_data = image_data.image.split(',')[1]

  # Decode base64 and load image
  decoded_image = base64.b64decode(base64_data)

  image_np = np.frombuffer(decoded_image, dtype=np.uint8)
  image = cv2.imdecode(image_np, flags=cv2.IMREAD_COLOR)

  results = model.predict(image, device="cpu", max_det=1, conf=0.55, imgsz=512, classes=letter_to_detect )
  names = model.names

  predicted_class = ""
  predictions = []
  for r in results:
      boxes = r.boxes
      labels = r.names  
      for c in r.boxes.cls:
          predicted_class = names[int(c)]
          predictions.append(predicted_class)

      for box, label in zip(boxes, labels):
          # Extract box coordinates
          x, y, w, h = box.xyxy[0].int().tolist()

          # Draw bounding box on the image
          cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

          # Display class label
          cv2.putText(image, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  imageKey = uuid.uuid4()
  output_path = os.path.join(saved_image_dir, f"{imageKey}.jpg")
  cv2.imwrite(output_path, image)
  saved_image = cv2.imread(output_path)
  _, encoded_image = cv2.imencode(".jpg", saved_image)
  base64_encoded_image = base64.b64encode(encoded_image).decode("utf-8")
  os.remove(output_path)
  responseData = ImageResponseData()
  if len(predictions) == 1 and predictions[0] == image_data.letter.upper():
      responseData.image = base64_encoded_image
      responseData.letterResult = 1
      return responseData
  else:
      responseData.image = base64_encoded_image
      responseData.letterResult = -1
  return responseData

