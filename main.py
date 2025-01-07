from fastapi import FastAPI, Request, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

import os
import warnings
import re
import logging
from datetime import datetime
import pprint
import itertools
import base64
import chardet
import markdown

import pandas as pd

import duckdb

from sqlalchemy import create_engine, inspect, event, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import DBAPIError

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Set the OpenAI API key in the environment variables.
# Replace <YOUR_KEY_HERE> with the actual API key string.
os.environ["OPENAI_API_KEY"] = "<YOUR_KEY_HERE>"

# Data Loading and Database Setup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("process.log"),
        logging.StreamHandler()
    ]
)

# Directory containing the data files
DATA_FOLDER = "./data"
UPLOAD_FOLDER = "./uploaded"
DATABASE_PATH = "HomeTask_PwC.duckdb"

# Create a persistent DuckDB engine
engine = create_engine(f"duckdb:///{DATABASE_PATH}")


def get_table_name(file_path):
    """
    Generate a sanitized table name based on the file name.

    Args:
        file_path (str): The file path of the input file.

    Returns:
        str: A sanitized table name.
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    sanitized_name = re.sub(r'[^a-zA-Z0-9]+', '_', base_name).strip('_')
    if not sanitized_name[0].isalpha():
        sanitized_name = f"table_{sanitized_name}"
    return sanitized_name


def read_file_to_dataframe(file_path):
    """
    Read a CSV or Excel file into a Pandas DataFrame.

    Args:
        file_path (str): The path to the input file.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    try:
        if file_path.endswith('.csv'):
            logging.info(f"Reading CSV file: {file_path}")
            try:
                return pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                with open(file_path, "rb") as f:
                    raw_data = f.read()
                    result = chardet.detect(raw_data)  # Detect file encoding
                    encoding = result["encoding"]
                return pd.read_csv(file_path, encoding=encoding) 
        elif file_path.endswith('.xlsx'):
            logging.info(f"Reading Excel file: {file_path}")
            return pd.read_excel(file_path, index_col=0)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}")
        return None


def write_dataframe_to_table(df, table_name):
    """
    Write a DataFrame to a DuckDB SQL table.

    Args:
        df (pd.DataFrame): The DataFrame to write.
        table_name (str): The name of the SQL table.

    Returns:
        None
    """
    try:
        df.to_sql(table_name, con=engine, if_exists="replace", index=True)
        logging.info(f"Data successfully written to the table: {table_name}")
    except Exception as e:
        logging.error(f"Failed to write DataFrame to table {table_name}: {e}")


def process_data_files(data_folder):
    """
    Process all CSV and Excel files in the specified folder.

    Args:
        data_folder (str): The folder containing the data files.

    Returns:
        None
    """
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        if file_name.endswith(('.csv', '.xlsx')):
            logging.info(f"Processing file: {file_name}")
            df = read_file_to_dataframe(file_path)
            if df is not None:
                table_name = get_table_name(file_path)
                write_dataframe_to_table(df, table_name)
        else:
            logging.warning(f"Skipping unsupported file: {file_name}")

logging.info("Starting data processing...")
if os.path.exists(DATA_FOLDER):
    process_data_files(DATA_FOLDER)
    logging.info("Data processing completed.")
else:
    logging.error(f"Data folder not found: {DATA_FOLDER}")

def save_and_load_file(upload_file: UploadFile):
    """
    Save the uploaded file and load its content into a DataFrame with encoding handling.
    """
    file_location = os.path.join(UPLOAD_FOLDER, upload_file.filename)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Save the file
    with open(file_location, "wb") as f:
        f.write(upload_file.file.read())

    # Detect encoding for CSV files
    if upload_file.filename.endswith(".csv"):
        with open(file_location, "rb") as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)  # Detect file encoding
            encoding = result["encoding"]

        if not encoding:
            raise ValueError("Unable to detect file encoding.")

        try:
            return pd.read_csv(file_location, encoding=encoding), file_location
        except Exception as e:
            raise ValueError(f"Error reading CSV with detected encoding '{encoding}': {e}")

    # Handle Excel files
    elif upload_file.filename.endswith(".xlsx"):
        try:
            return pd.read_excel(file_location), file_location
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

    else:
        raise ValueError("Unsupported file format.")

# Define and enforce a rule prohibiting DML statements (DROP, INSERT, UPDATE, DELETE, MERGE)
# by listening for execution events.
# If a forbidden statement is detected, raise an error before cursor execution.

 # List of forbidden keywords
FORBIDDEN_KEYWORDS = ["drop", "delete", "insert", "update", "merge"]

def validate_no_dml(engine, clause, multiparams, params):
    # Extract the SQL statement
    statement = str(clause).strip().lower()

    # Check if the statement starts with forbidden keywords
    if any(statement.startswith(keyword) for keyword in FORBIDDEN_KEYWORDS):
        raise DBAPIError("Nice try ;), but DML statements are not allowed (DROP, DELETE, INSERT, UPDATE, MERGE).", None, None)

 # Attach the event listener to the engine
event.listen(engine, "before_execute", validate_no_dml)



# Create a multi-line prompt template for ChatGPT using the COSTAR framework.
# Include additional instructions and a few-shot example section 
# to guide ChatGPT in responding with concise, SQL-driven insights.

template = '''
Context: The department responsible for data quality management requires a solution for data profiling to allow non-technical users to ask simple questions about their data, such as identifying empty fields or detecting outliers. The goal is to enable interaction with data through plain language queries and return concise, actionable answers based on SQL queries executed on the database.
Objective: Provide accurate, brief, and clear answers to user queries regarding data analytics and quality, utilizing SQL to generate insights.
Scope: Focus on questions related to data profiling, such as:
 - Counts of missing or empty fields.
 - Detection of outliers.
 - Basic descriptive statistics (e.g., min, max, mode, median, averages, distributions).
Target Audience: Data analysts and non-technical users requiring simplified and accessible insights.
Approach: Design responses to be:
 - SQL-driven, based on direct database queries.
 - Short, straightforward, and easily understood by non-technical users.
 - Written in a professional yet approachable tone.
Result: A clear and concise answers to user questions, ensuring they align with the technical accuracy of SQL-based insights while being accessible to non-technical users.

You are an agent designed to interact with a SQL database.

Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP, MERGE etc.) to the database.

If requested, Generate an SQL query to detect anomalies or outliers in a dataset using the 1.5 IQR (Interquartile Range) method.

To start you should ALWAYS look at the tables in the database to see what you can query. Do NOT skip this step.
Then you should query the schema of the most relevant tables.

Below are a number of examples of questions and their corresponding SQL queries:

  "input": "List all Transaction Type.",
  "query": "SELECT DISTINCT \"Bus. Transac. Type\" FROM TABLE;"

  "input": "Find all Transactions made in 2005 fiscal year.",
  "query": "SELECT * FROM \"TABLE\" WHERE \"Fiscal Year.1\" = 2005;"

  "input": "Give me all Transactions with maximum value.",
  "query": "SELECT * FROM \"TABLE\" WHERE ABS(\"Transaction Value\") = (SELECT MAX(ABS(\"Transaction Value\")) FROM \"[table_name]\");"

"input": "if there are any outliers present in Transaction Value?", # "Give me outliers in the [Transaction Value] column",
"query": """
    WITH Ordered_Values AS (
        SELECT [Transaction Value]
        FROM TABLE
        ORDER BY [Transaction Value]
    ), 
    Row_Numbers AS (
        SELECT 
            [Transaction Value],
            ROW_NUMBER() OVER (ORDER BY [Transaction Value]) AS RowNum,
            COUNT(*) OVER () AS TotalRows
        FROM Ordered_Values
    ), 
    Quartiles AS (
        SELECT 
            MIN(CASE WHEN RowNum = CAST(TotalRows * 0.25 AS INT) THEN [Transaction Value] END) AS Q1,
            MIN(CASE WHEN RowNum = CAST(TotalRows * 0.75 AS INT) THEN [Transaction Value] END) AS Q3
        FROM Row_Numbers
    ), 
    IQR_Calculations AS (
        SELECT 
            Q1,
            Q3,
            (Q3 - Q1) AS IQR,
            Q1 - 1.5 * (Q3 - Q1) AS Lower_Bound,
            Q3 + 1.5 * (Q3 - Q1) AS Upper_Bound
        FROM Quartiles
    )
    SELECT 
        SUM(CASE WHEN [Transaction Value] < Lower_Bound THEN 1 ELSE 0 END) AS Transactions_Below_Lower_Bound,
        SUM(CASE WHEN [Transaction Value] > Upper_Bound THEN 1 ELSE 0 END) AS Transactions_Above_Upper_Bound,
        SUM(CASE WHEN [Transaction Value] < Lower_Bound OR [Transaction Value] > Upper_Bound THEN 1 ELSE 0 END) AS Total_Outliers
    FROM 
        "[table_name]",
        IQR_Calculations;      
        """

"input": "Give me descriptive/basic statistics for the [column_name]",
"query": """        
    WITH basic_stats AS (
        SELECT
            COUNT("[column_name]") AS count_val,
            MIN("[column_name]")   AS min_val,
            MAX("[column_name]")   AS max_val,
            AVG("[column_name]")   AS mean_val
        FROM "table_name"
    ),
    variance_calc AS (
        SELECT
            -- Sample variance with Bessel's correction
            SUM(
                ("[column_name]" - (SELECT mean_val FROM basic_stats)) *
                ("[column_name]" - (SELECT mean_val FROM basic_stats))
            ) / ((SELECT count_val FROM basic_stats) - 1) AS variance_val
        FROM "table_name"
    ),
    quartiles AS (
        SELECT
            QUANTILE("[column_name]", 0.25) AS Q1,
            QUANTILE("[column_name]", 0.50) AS Median,
            QUANTILE("[column_name]", 0.75) AS Q3
        FROM "[table_name]"
    )
    SELECT
        (SELECT count_val FROM basic_stats) AS "Count",
        (SELECT min_val   FROM basic_stats) AS "Min",
        (SELECT max_val   FROM basic_stats) AS "Max",
        (SELECT mean_val  FROM basic_stats) AS "Mean",
        ROUND(SQRT((SELECT variance_val FROM variance_calc)), 3)       AS "StdDev",   -- optional rounding
        (SELECT Q1 FROM quartiles)         AS Q1,
        (SELECT Median FROM quartiles)     AS Median,
        (SELECT Q3 FROM quartiles)         AS Q3;
"""

  "input": "Count null and non-null values in column [column_name]",
  "query": "SELECT COUNT(*) - COUNT(\"[column_name]\") AS COUNT_NULLS, COUNT(\"[column_name]\") AS COUNT_NOT_NULLS FROM \"[table_name]\";"
'''

# Create a PromptTemplate instance from the multi-line prompt string.
prompt_template = PromptTemplate.from_template(template)

# Format the prompt template, providing SQLite as the SQL dialect 
# and specifying top_k, which could represent the number of records to retrieve.

system_message = prompt_template.format(dialect="postgresql", top_k=5)


config = {"configurable": {"thread_id": "1"}}


# We can add "chat memory" to the graph with LangGraph's checkpointer
# to retain the chat context between interactions
memory = MemorySaver()


app = FastAPI()

# Mount the 'static' directory
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/get-chat-bot-response")
def get_bot_response(msg: str):

    if msg.lower().startswith("ping"):
        return 'Pong'

    # Building the Question/Answering LLM System

    # Check tables detected by SQLAlchemy
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    # Initialize an langchain SQLDatabase instance with the existing engine to handle queries.
    sql_database = SQLDatabase(engine, include_tables=tables)

    # Initialize the ChatOpenAI "gpt-4o-mini" model with a specified temperature
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # create a toolkit with the given database,
    toolkit = SQLDatabaseToolkit(db=sql_database, llm=llm)

    # retrieve the associated tools from the toolkit
    tools = toolkit.get_tools()

    # Initialize a prebuilt LangGraph React agent with the formatted state_modifier.
    agent_executor = create_react_agent(llm, tools, state_modifier=system_message, checkpointer=memory)

    logging.info(f"User question: {msg}")

    try:
        # Stream responses from the React Agent step by step by calling it
        step_answers = []
        for step in agent_executor.stream(
            {"messages": [{"role": "user", "content": msg}]},
            stream_mode="values",
            config=config
        ):
            step_answers.append(step["messages"][-1].content)
                
        # Retrieve the final answer and add it to the chat history.
        answer = step_answers[-1]
        logging.info(f"Agent answer: {answer}")
    except Exception as e:
        # Handle exceptions by displaying an error message.
        answer = "<b>Error:</b> " + str(e)
        logging.info("<b>Error:</b> " + str(e))
     
    answer_formatted = markdown.markdown(answer.replace("$", r"\$"))  


    return str(answer_formatted)


@app.get("/api/get-tables")
async def get_tables():
    """
    Return a list of all tables in the DuckDB database.
    """
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        return {"tables": tables}  # Ensure the response is a dictionary with a `tables` key
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-table-sample")
async def get_table_sample(
    table_name: str,
    limit: int = Query(10, ge=1, le=100, description="Number of rows to fetch (default: 10)"),
):
    """
    Fetch a sample of data from a given table dynamically without pre-knowledge of fields.
    """
    try:
        inspector = inspect(engine)
        # Verify the table exists
#        inspector = inspect(engine)
        if table_name not in inspector.get_table_names():
            raise HTTPException(status_code=404, detail=f"Table '{table_name}' not found.")

        # Dynamically query the table
        query = text(f"SELECT * FROM {table_name} LIMIT :limit")
        with engine.connect() as connection:

            result = connection.execute(query, {"limit": limit})
            rows = [dict(row._mapping) for row in result]  # Use _mapping for row to dict conversion

        return {"table_name": table_name, "sample": rows}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/get-fields")
async def get_fields(table: str = Query(..., description="Name of the table to fetch fields from")):
    """
    Return a list of fields (columns) for a given table in the DuckDB database.
    """
    try:
        # Use SQLAlchemy's inspector to get table metadata
        inspector = inspect(engine)
        
        # Check if the table exists
        if table not in inspector.get_table_names():
            raise HTTPException(status_code=404, detail=f"Table '{table}' not found.")
        
        # Get column names for the table
        columns = [column["name"] for column in inspector.get_columns(table)]
        
        # Return the list of fields
        return {"fields": columns}

    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

# File Upload Endpoint
@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):

    try:
        if not file.filename.endswith((".csv", ".xlsx")):
            raise HTTPException(status_code=400, detail="Invalid file format. Only .csv and .xlsx are supported.")

        # Save and process the file
        df, file_path = save_and_load_file(file)

        # Remove the validate_no_dml event listener.
        event.remove(engine, "before_execute", validate_no_dml)

        process_data_files(UPLOAD_FOLDER)

        # Re-attach the validate_no_dml event listener.
        event.listen(engine, "before_execute", validate_no_dml)

        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"File '{file_path}' has been deleted.")
            else:
                print(f"File '{file_path}' does not exist.")

        except Exception as e:
            print(f"An error occurred while deleting the file: {e}")

        return JSONResponse(content={"message": "File uploaded successfully"})

    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app")
