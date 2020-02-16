FROM python:3.7-slim

WORKDIR /app
COPY . /app

RUN pip install -r req.txt 

EXPOSE 5000

CMD [ "python3","app.py"]