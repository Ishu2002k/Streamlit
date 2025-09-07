# step_3.py
import sqlparse
import os
import sqlitecloud
import pandas as pd
import plotly.express as px
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from step_1 import extract_schema, generate_embedding_text,fetch_table_name
from openai import AzureOpenAI
from sql_txt_2 import (
    generate_embeddings_hf,
    build_schema_kg,
    find_join_path,
    build_faiss_vectorstore
)

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
        model=LLM_MODEL,
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

# ----------- Visualization ----------------
def visualize_results(df: pd.DataFrame):
    """Simple heuristic: categorical vs numerical → bar chart / line chart"""
    if df.empty:
        print("⚠️ No results to visualize.")
        return
    
    if len(df.columns) >= 2:
        x, y = df.columns[:2]
        if pd.api.types.is_numeric_dtype(df[y]):
            fig = px.bar(df, x=x, y=y, title="SQL Query Results")
        else:
            fig = px.histogram(df, x=x, title="SQL Query Results")
        fig.show()
    else:
        print(df)

# ----------------- KG VALIDATION -----------------
def validate_sql_with_kg(sql_query, kg):
    parsed = sqlparse.parse(sql_query)
    if not parsed:
        return False, ["❌ SQL parsing failed"]

    stmt = parsed[0]
    tokens = [t for t in stmt.tokens if not t.is_whitespace]

    used_tables, used_columns = set(), set()
    hints = []

    for token in tokens:
        tval = token.value.strip().replace("`", "").replace('"', "")
        if "." in tval:  
            used_columns.add(tval)
            table = tval.split(".")[0]
            used_tables.add(table)
        else:
            if tval.upper() not in ["SELECT", "FROM", "WHERE", "JOIN", "ON",
                                    "GROUP", "BY", "ORDER", "LIMIT", "AND", "OR"]:
                used_tables.add(tval)

    # KG tables & columns
    kg_tables = {n for n, d in kg.nodes(data=True) if d.get("type") == "table"}
    kg_columns = {n for n, d in kg.nodes(data=True) if d.get("type") == "column"}

    # Validate columns
    for c in used_columns:
        if c not in kg_columns:
            table, col = c.split(".")[0], c.split(".")[1]
            if table in kg_tables:
                hints.append(
                    f"Column '{col}' not found in {table}. "
                    f"Suggested join path: {find_join_path(kg, table, 'People')}"
                )
            else:
                hints.append(f"Column '{c}' not found in KG.")

    return (len(hints) == 0), hints

# ----------- Hybrid RAG + KG Pipeline ----------------
def answer_query(nl_query, vectorstore, kg):
    # Step 1: Retrieve semantic context from FAISS/Chroma
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(nl_query)
    semantic_context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Step 2: Check for joins using Knowledge Graph
    join_hints = []
    tables = [n for n, d in kg.nodes(data=True) if d.get("type") == "table"]
    for i, t1 in enumerate(tables):
        for t2 in tables[i+1:]:
            path = find_join_path(kg, t1.strip('"'), t2.strip('"'))
            if path:
                join_hints.append(f"Join path between {t1} and {t2}: {path}")

    # Step 3: Prompt LLM with hybrid context
    system_prompt = """
        You are an expert SQL query generator.
        Use ONLY these table names exactly as given:
        {fetch_table_name(db_path)}
        Generate a **SELECT-only** SQL query for SQLite. 
        Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE.
        Generate valid SQL query for the user query.
        Use provided schema context and relationships.
        Take care of values in the table irrespective user asked in capital or in lower case.
        Do not explain. No backticks.
        Properly which column is present in which table.
        Do not assign any column name to any table.
        Only return SQL, nothing else.
    """

    user_prompt = f"""
    Natural language query:
    {nl_query}

    Schema + semantic context:
    {semantic_context}

    Possible join paths (from Knowledge Graph):
    {join_hints}
    """

    sql_query = llm_complete(system_prompt, user_prompt)

    print("\n✅ Generated SQL:\n", sql_query)

    # Step 4: KG Validation
    is_valid, hints = validate_sql_with_kg(sql_query, kg)

    # Step 5: Execute SQL
    if not is_valid:
        print("\n⚠️ KG Validation failed with hints:", hints)

        correction_prompt = f"""
        Use Only these tables name:
        {fetch_table_name(db_path)}
        The previous SQL had schema issues:
        {hints}

        Please regenerate a corrected SQL query for:
        {nl_query}

        Use schema + KG hints correctly this time.
        """

        sql_query = llm_complete(system_prompt, correction_prompt)
        print("\n✅ Corrected SQL:\n", sql_query)

    df = run_sql_query(sql_query)
    print("\n📊 Query Results:\n", df)

    # Step 5: Visualization
    # visualize_results(df)
    # return df

# ------------------- MAIN --------------------
if __name__ == "__main__":
    # Extract schema
    schema_info = extract_schema(db_path)

    # Embeddings
    embedding_docs = generate_embedding_text(schema_info)
    embedded_docs = generate_embeddings_hf(embedding_docs)

    # Vectorstore (FAISS or Chroma)
    vectorstore = build_faiss_vectorstore(embedded_docs)
    # vectorstore = build_chroma_vectorstore(embedded_docs)  # persistent option

    # Build Knowledge Graph
    G = build_schema_kg(schema_info)

    # Example NL query
    nl_query = "Find all the People with region they belong who have returned atleast one order"
    answer_query(nl_query, vectorstore, G)
