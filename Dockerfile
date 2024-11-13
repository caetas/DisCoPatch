FROM python:3.11-slim

WORKDIR /app/

COPY requirements/requirements.txt /app/requirements/
RUN pip install -r /app/requirements/requirements.txt

# copy code and models
#ADD models /app/models
ADD src /app/src
WORKDIR /app

COPY .env /app/.env

ENV PYTHONPATH="${PYTHONPATH}:/app/src/patchnorm"

# The code to run when container is started
CMD python src/patchnorm/api.py
