# For setting up, running, and testing the solution

### 1. Run PoC with Jupyter Notebook

```bash
jupyter notebook
```
Run PoC_QA_LLM_DuckDB.ipynb

### 2. Run Locally with FastAPI Application

```bash
uvicorn main:app --reload
```
Check http://127.0.0.1:8000

### 3. Deploy to Heroku

```bash
heroku login
git init
heroku git:remote -a ai-chatbot-test
git add .
git commit -am "initial commit"
it push heroku master
```
