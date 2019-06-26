FROM python:3.7
ENV PYTHONUNBUFFERED 1
WORKDIR /home/4aibd-s1-project-ml

# Add requirements file 
ADD requirements.txt /app/
RUN pip install -r /app/requirements.txt
EXPOSE 5000

#RUN curl -sSL https://sdk.cloud.google.com | bash
#ENV PATH $PATH:/root/google-cloud-sdk/bin
