FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip -r install requirements.txt

COPY app.py app.py
COPY templates templates

EXPOSE 8080

CMD ["python", "app.py"]
