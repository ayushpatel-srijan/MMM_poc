FROM python:3.9-slim-buster

RUN apt-get update
RUN python -m pip install --upgrade pip
RUN apt-get install git -y

WORKDIR /app

COPY requirements.txt .
RUN apt-get install build-essential
RUN pip install --upgrade setuptools wheel
RUN pip install -r requirements.txt
RUN pip install git+https://github.com/stan-dev/pystan2.git@master

EXPOSE 8502

COPY . .

CMD ["streamlit", "run","app.py","--server.port=8502"]
