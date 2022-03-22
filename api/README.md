# api

Install deps:
```bash
git clone git@github.com:oxfordeo/oxeo-water.git
cd oxeo-water
pip install -e .
cd api
pip install -r requirements.txt
```

Run the API:
```bash
export PG_DB_USER=reader
export PG_DB_PW=waterandbasinsreadonly
export PG_DB_HOST=35.204.253.189
export GOOGLE_APPLICATION_CREDENTIALS=../../keys/oxeo-main-prefect.json
uvicorn app.main:app --reload
```
