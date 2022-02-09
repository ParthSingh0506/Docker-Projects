FROM python:latest

WORKDIR /root/Projects/Docker

COPY . .

RUN pip install --upgrade pip
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# command that runs when container starts
CMD ["python" , "/root/Projects/Docker/ML.py"]