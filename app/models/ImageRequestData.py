from pydantic import BaseModel

class ImageRequestData(BaseModel):
    image: str
    letter: str