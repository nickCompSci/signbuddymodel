from pydantic import BaseModel

class ImageResponseData(BaseModel):
    image: str = "None"
    letterResult: int = -1