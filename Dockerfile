FROM python:3.6

WORKDIR /app

COPY . .

RUN pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.12.0-py3-none-any.whl && \
  pip install -r requirements.txt

ENTRYPOINT ["python", "server.py"]
