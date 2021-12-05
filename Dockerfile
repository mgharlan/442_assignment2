FROM jupyter/pyspark-notebook:latest

WORKDIR /442_assignment2

COPY . /442_assignment2

ENV PYSPARK_PYTHON=python3

ENV PYSPARK_DRIVER_PYTHON=python3

RUN pip install findspark

CMD ["python3", "src/main.py"]
