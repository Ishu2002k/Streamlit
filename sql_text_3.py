# step_3.py
import os
import sqlitecloud
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

# -------- Load env --------
load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# ---------- Azure OpenAI Client ----------
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

LLM_MODEL = os.getenv("LLM_MODEL")

def llm_complete(system: str, user: str, temperature=0):
    resp = client.chat.completions.create(
        model = LLM_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    sql = resp.choices[0].message.content.strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql

# ----------- Query Runner ----------------
def run_sql_query(query: str):
    """Run SQL query on SQLite cloud db and return dataframe"""
    con = sqlitecloud.connect(db_path)
    df = pd.read_sql_query(query, con)
    con.close()
    return df