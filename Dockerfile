FROM python:3.9-slim-buster

RUN apt-get update
RUN python -m pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["streamlit", "run", "--server.port", "8502", "app.py"]
