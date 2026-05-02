FROM python:3.11-slim

WORKDIR /app

COPY Flask/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Flask/ .

EXPOSE 5000

CMD ["python", "app.py"]