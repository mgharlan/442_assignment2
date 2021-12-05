FROM python:3.9.9-slim

WORKDIR /442_assignment2

COPY . /442_assignment2

RUN pip install -r requirements.txt

CMD ["python", "src/main.py"]