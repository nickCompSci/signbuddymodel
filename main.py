
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import base64
import cv2
from ultralytics import YOLO
import numpy as np
import time

app = FastAPI()

# Load the YOLOv8 ONNX model
# onnx_model_path = "./models/best.onnx"
# ort_session = onnxruntime.InferenceSession(onnx_model_path)

origins = [
    "http://localhost:5173",
]
alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image: str
    letter: str

# Load YOLOv8 model
model_path = "./models/best.pt"
model = YOLO('./models/best.pt')  # load a custom trained model

@app.post("/verifyimage")
async def process_image(image_data: ImageData):
  start_time = time.time()
  base64_data = image_data.image.split(',')[1]

# Decode base64 and load image
  decoded_image = base64.b64decode(base64_data)

  with open("./images/imageToSave.jpg", "wb") as fh:
      fh.write(decoded_image)

  image = cv2.imread('./images/imageToSave.jpg')
  image = image.astype(np.uint8)

  results = model.predict(image, max_det=1, conf=0.65, imgsz=512)
  names = model.names
  # print(names)
  predicted_class = ""
  speeds = {}
  for r in results:
      # print(r)
      boxes = r.boxes
      labels = r.names  #pp Get class labels
      speeds = r.speed
      # print(labels)
      for c in r.boxes.cls:
          predicted_class = names[int(c)]

      for box, label in zip(boxes, labels):
          # print(label)
          # Extract box coordinates
          x, y, w, h = box.xyxy[0].int().tolist()

          # Draw bounding box on the image
          cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

          # Display class label
          # print(label)
          label_str = str(labels[label])
          cv2.putText(image, predicted_class, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
  cv2.imshow('Image with Bounding Boxes and Class Labels', image)
  output_path = './images/predicted.jpg'  # Provide the desired path
  cv2.imwrite(output_path, image)
  end_time = time.time()

  # Calculate the elapsed time
  elapsed_time = end_time - start_time
  print(f"Object detection took {elapsed_time:.4f} seconds")

  return f"Model predicted class: {predicted_class} with speeds: {speeds}"