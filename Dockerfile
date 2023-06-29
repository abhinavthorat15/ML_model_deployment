FROM python:3.8-slim-buster
WORKDIR /app
COPY . /app
EXPOSE 5000

RUN apt update -y && apt install awscli -y

RUN pip install -r requirements.txt
CMD python app.py

