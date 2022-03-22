# syntax=docker/dockerfile:1

FROM eu.gcr.io/oxeo-main/python-gcloud-auth:latest

COPY oxeo/ water/oxeo/
COPY pyproject.toml setup.cfg setup.py water/
RUN pip install -e ./water/

COPY api/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY api api
WORKDIR api
RUN ls .

ENV PORT 8080
EXPOSE $PORT

CMD uvicorn app.main:app --host=0.0.0.0 --port=$PORT
