FROM python:3.7
ENV PYTHONUNBUFFERED 1
WORKDIR /home/4aibd-s1-project-ml
# Add requirements file 
ADD requirements.txt /app/

RUN pip install -r /app/requirements.txt

EXPOSE 5000

CMD mlflow server \
    --host 0.0.0.0 \
    --port=5000
