import os
import sqlite3
import sqlitecloud
import json
import logging
from dotenv import load_dotenv
from openai import AzureOpenAI
import sqlglot
from datetime import datetime

# Step 2 outputs
from step_1 import extract_schema, generate_embedding_text
from step_2 import build_chroma_vectorstore, build_knowledge_graph, generate_embeddings_hf 
from step_3 import search_schema

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DATABASE_CONNECTION_STRING")
LOG_FILE = "nl_to_sql_feedback.log"

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(message)s')

client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# -----------------------------
# Helper functions
# -----------------------------
def reasoning_node(user_query, context_text):
    """
    Convert natural language query + schema context to structured reasoning plan.
    """
    prompt = f"""
        You are an expert SQL assistant. 
        Given the following schema context:

        {context_text}

        And the user query:

        {user_query}

        Use exact table and column names as in the schema.
        Ensure all string filters are **case-insensitive**, e.g., using LOWER(column) = LOWER(value).
        Output a step-by-step reasoning plan for how to construct the SQL, 
        including which tables, joins, groupings, and filters are required. 
        Return as JSON with keys: tables, joins, filters, group_by, order_by, limit.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    plan_text = response.choices[0].message.content
    print(f"plan_text: {plan_text}")

    try:
        plan_json = json.loads(plan_text)
    except:
        plan_json = {"raw_plan": plan_text}  # fallback
    return plan_json

def sql_generation_node(reasoning_plan, context_text, user_query):
    """
    Generate SQL using LLM from reasoning plan + schema context.
    All string comparisons are made case-insensitive using LOWER().
    """
    prompt = f"""
        You are an expert SQL generator. Only return the sql query.
        Schema context:

        {context_text}

        User query:

        {user_query}

        Reasoning plan:

        {reasoning_plan}

        Generate a **SELECT-only** SQL query. 
        All string comparisons should be **case-insensitive** using LOWER(column) = LOWER('value').
        Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE.
        Generate valid SQL query for the user query.
        """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    sql_query = response.choices[0].message.content.strip()
    # Strip code fences if the model adds them
    sql = sql_query.replace("```sql", "").replace("```", "").strip()
    print(F"\n\nsql: {sql} \n\n")
    return sql

def validation_node(sql_query):
    """
    Validate SQL using sqlglot. Returns corrected SQL if possible.
    """
    print(sql_query)
    try:
        parsed = sqlglot.parse_one(sql_query)
        # optional: format SQL
        return parsed.sql(),None
    except Exception as e:
        return None, str(e)

def execution_node(sql_query, db_path=DB_PATH):
    """
    Execute SQL on SQLite DB and return results.
    """
    # conn = sqlite3.connect(db_path)
    conn = sqlitecloud.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        return columns, rows
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

def feedback_log(user_query, sql_query, result, error=None):
    """
    Log user query, generated SQL, result, and errors for later analysis.
    """
    if result is not None and isinstance(result, dict):
        if "rows" in result:
            result["rows"] = [list(r) for r in result["rows"]]  # convert each row tuple to list
    
    log_entry = {
        "timestamp": str(datetime.now()),
        "user_query": user_query,
        "generated_sql": sql_query,
        "result": result,
        "error": error
    }
    logging.info(json.dumps(log_entry))

# -----------------------------
# LangGraph Orchestration
# -----------------------------
def langgraph_pipeline(user_query, vectorstore, kg_graph, top_k=5):
    # Step 1: Schema retrieval
    context_text, involved_tables = search_schema(user_query, vectorstore, kg_graph, top_k=top_k)
    
    # Step 2: Reasoning node
    reasoning_plan = reasoning_node(user_query, context_text)
    
    # Step 3: SQL generation node
    sql_query = sql_generation_node(reasoning_plan, context_text, user_query)
    
    # Step 4: Validation node
    validated_sql, validation_error = validation_node(sql_query)
    if validated_sql is None:
        validation_error = "SQL validation failed"
        feedback_log(user_query, sql_query, None, validation_error)
        return None, None, validation_error
    
    # Step 5: Execution node
    columns, rows_or_error = execution_node(validated_sql)
    if columns is None:
        feedback_log(user_query, validated_sql, None, rows_or_error)
        return None, None, rows_or_error
    
    # Step 6: Feedback log
    feedback_log(user_query, validated_sql, {"columns": columns, "rows": rows_or_error})
    
    return validated_sql, {"columns": columns, "rows": rows_or_error}, None

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Step 0: Prepare vector store + KG (from Step 2)
    schema_info = extract_schema(DB_PATH)
    embedding_docs = generate_embedding_text(schema_info)
    embedding_docs = generate_embeddings_hf(embedding_docs)
    vectorstore = build_chroma_vectorstore(embedding_docs)
    kg_graph = build_knowledge_graph(schema_info)
    
    # Example user query
    user_query = "Find person details with total amount he spend on purchasing after 4th January 2023"
    
    sql, result, error = langgraph_pipeline(user_query, vectorstore, kg_graph)
    
    if error:
        print("Error:", error)
    else:
        print("--- Generated SQL ---")
        print(sql)
        print("--- Query Result ---")
        print(result)
