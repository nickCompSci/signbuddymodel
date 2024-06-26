# syntax = docker/dockerfile:1.2

#
FROM python:3.9

#
WORKDIR /code

RUN apt-get update
RUN apt install -y libgl1-mesa-glx
#
COPY ./requirements.txt /code/requirements.txt

#
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]