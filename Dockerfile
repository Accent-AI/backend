FROM python:3.8-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y ffmpeg git && apt-get clean

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
