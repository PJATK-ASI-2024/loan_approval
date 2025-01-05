
FROM python:3.12
WORKDIR /app
RUN pip install --upgrade pip
RUN pip install pandas fastapi uvicorn
COPY . .
EXPOSE 5000
CMD ["uvicorn", "server:app", "--reload", "--port=5000", "--host=0.0.0.0"]