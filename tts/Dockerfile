FROM python:3.8-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY install.sh install.sh

RUN chmod +x ./install.sh && ./install.sh

COPY requirements.txt requirements.txt

RUN python3 -m pip install --upgrade setuptools wheel      
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python3", "main.py" ]
