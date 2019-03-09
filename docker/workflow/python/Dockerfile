FROM python:3.7
ENV PYTHONUNBUFFERED 1
WORKDIR /home/4aibd-s1-project-ml
# Add requirements file 
ADD requirements.txt /app/

# workflow needs a home, ~/workflow is the default,
# but you can lay foundation somewhere else if you prefer
# (optional)
RUN export AIRFLOW_HOME=~/airflow

RUN pip install -r /app/requirements.txt

# initialize the database
#RUN python airflow initdb

# start the web server, default port is 8080
# RUN python airflow webserver -p 8080

# start the scheduler
# RUN python airflow scheduler