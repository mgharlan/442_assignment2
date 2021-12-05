FROM python:3.9.9-slim

WORKDIR /442_assignment2

COPY . /442_assignment2

RUN python -m venv .

RUN source bin/activate

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD ["python", "src/main.py"]