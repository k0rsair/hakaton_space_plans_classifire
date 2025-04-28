FROM python:3.9-slim

WORKDIR /python

# Установка системных библиотек
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    libblas-dev \
    liblapack-dev \
    python3-dev \
    python3-pip

COPY ./python .

# Установка зависимостей Python
RUN pip install --upgrade pip
RUN pip install -v --no-cache-dir -r ./requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]