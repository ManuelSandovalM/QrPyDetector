FROM python:3.10

RUN pip install --upgrade pip 

WORKDIR /code 

RUN pip install fastapi uvicorn pillow tensorflow  

COPY . /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]