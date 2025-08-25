import os
import re
import sqlite3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
import sqlitecloud  # Use sqlitecloud

# Load .env file
load_dotenv()

# -----------------------------
# Azure OpenAI client
# -----------------------------
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# -----------------------------
# Helpers
# -----------------------------
FORBIDDEN_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|ATTACH|REINDEX|VACUUM)\b", re.IGNORECASE)

def load_schema(conn: sqlite3.Connection) -> tuple[str, dict]:
    """
    Returns:
      schema_str: human-readable schema text for prompting
      schema_map: dict {table_name: [(col_name, col_type), ...]}
    """
    tables = pd.read_sql(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' and name NOT LIKE '_sqlite%';",
        conn
    )["name"].tolist()

    schema_parts = []
    schema_map = {}

    for table in tables:
        df_info = pd.read_sql(f"PRAGMA table_info({table});", conn)
        cols = [(r["name"], r["type"]) for _, r in df_info.iterrows()]
        schema_map[table] = cols
        cols_text = ", ".join([f"{c} ({t or 'TEXT'})" for c, t in cols])
        schema_parts.append(f"Table: {table}\nColumns: {cols_text}")

    schema_str = "\n\n".join(schema_parts) if schema_parts else "No user tables found."
    return schema_str, schema_map

def summarize_schema(schema_map: dict) -> str:
    """
    Lightweight NL summary (no extra LLM call).
    """
    lines = []
    for t, cols in schema_map.items():
        colnames = [c for c, _ in cols]
        key_hints = [c for c in colnames if c.lower().endswith("_id") or c.lower() in {"id", "patient_id", "physician_id"}]
        lines.append(
            f"- {t}: {len(cols)} columns; possible keys: {', '.join(key_hints) if key_hints else 'n/a'}."
        )
    return "\n".join(lines)

FEW_SHOT = """
        Examples (SQLite):

        User: Count unique patients by region where age > 18
        SQL:
        SELECT pd.Region, COUNT(DISTINCT p.Patient_ID) AS treated_patients
        FROM SQL_103_Assessment_Set1_Data_Claims_Data c
        JOIN SQL_103_Assessment_Set1_Data_Patient_Demographics p ON p.Patient_ID = c.Patient_ID
        JOIN SQL_103_Assessment_Set1_Data_Physician_Demographics pd ON pd.Physician_ID = c.Physician_ID
        WHERE p.Age > 18
        GROUP BY pd.Region;

        User: List 10 most recent claims with physician specialty
        SQL:
        SELECT c.Claim_ID, c.Date, c.Patient_ID, d.Specialty
        FROM SQL_103_Assessment_Set1_Data_Claims_Data c
        JOIN SQL_103_Assessment_Set1_Data_Physician_Demographics d
          ON d.Physician_ID = c.Physician_ID
        ORDER BY c.Date DESC
        LIMIT 10;
        """

def build_generation_prompt(nl_request: str, schema_text: str, schema_summary: str) -> str:
    return f"""
        You are an expert SQL assistant. Output ONLY a valid SQLite SELECT statement, nothing else.
        Constraints:
        - Only SELECT queries are allowed.
        - Use JOINs when needed.
        - Use explicit table names from the schema.
        - If grouping/aggregation is required, ensure proper GROUP BY.
        - Do not include backticks or markdown fences.
        - Avoid non-SQL commentary.
        - Make conservative assumptions if names are ambiguous.

        Database schema:
        {schema_text}

        Schema summary:
        {schema_summary}

        {FEW_SHOT}

        Natural language request:
        {nl_request}

        Return only the SQL SELECT query.
        """

def build_correction_prompt(original_sql: str, error_msg: str, schema_text: str, nl_request: str, schema_summary: str) -> str:
    return f"""
        The following SQLite SELECT query failed. Fix it and return ONLY a corrected SELECT query.

        Original request:
        {nl_request}

        Schema:
        {schema_text}

        Schema summary:
        {schema_summary}

        Original SQL:
        {original_sql}

        Error:
        {error_msg}

        Rules:
        - Return only a valid SQLite SELECT query (no DDL/DML).
        - Use tables/columns that exist.
        - No markdown code fences.
        """

def call_llm_sql(prompt: str, temperature: float = 0.0, model: str = "gpt-4o-mini") -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You generate safe, strictly-SELECT SQLite queries only."},
            {"role": "user", "content": prompt},
        ],
    )
    sql = resp.choices[0].message.content.strip()
    # Strip code fences if the model adds them
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql

def enforce_select_only(sql: str) -> tuple[bool, str]:
    """
    Returns (ok, message). ok=False means block execution.
    """
    if not sql.strip().lower().startswith("select"):
        return False, "Only SELECT queries are allowed."
    if FORBIDDEN_SQL.search(sql):
        return False, "Detected a non-SELECT operation. Only SELECT is allowed."
    return True, ""

def extract_tables_from_sql(sql: str) -> list[str]:
    """
    Very simple heuristic to extract table names after FROM / JOIN.
    """
    candidates = []
    # get tokens following FROM or JOIN up to whitespace or punctuation
    for kw in [" from ", "\nfrom ", " join ", "\njoin "]:
        parts = re.split(kw, sql, flags=re.IGNORECASE)
        if len(parts) > 1:
            for p in parts[1:]:
                token = re.split(r"[\s\(\),;]+", p.strip())[0]
                if token and token.upper() not in {"SELECT"}:
                    candidates.append(token)
    # Deduplicate preserving order
    seen = set()
    ordered = []
    for t in candidates:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def safe_run_sql(conn: sqlite3.Connection, sql: str) -> tuple[pd.DataFrame | None, str | None]:
    try:
        df = pd.read_sql(sql, conn)
        return df, None
    except Exception as e:
        return None, str(e)

# -----------------------------
# Streamlit UI
# -----------------------------
def user_panel(use_cloud: bool = True):
    st.title("📊 SQL Query Assistant")

    # Session state
    if "history" not in st.session_state:
        st.session_state.history = []
    if "show_history" not in st.session_state:
        st.session_state.show_history = False

    # Sidebar
    show_history = st.sidebar.toggle(
        "Query History",
        # value=st.session_state.show_history,
        value=st.session_state.get("show_history", False),
        key="show_history",
        help="Toggle to view previous queries & results",
    )
    temperature = st.sidebar.slider("Model temperature", 0.0, 1.0, 0.0, 0.1)
    # model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-transcribe"], index=0)
    model_name = "gpt-4o-mini"

    # Database path (use the same DB the admin writes to)
    # db_path = st.sidebar.text_input("SQLite DB file", value="uploaded_db.sqlite", help="Path to your SQLite database file")
    # if not os.path.exists(db_path):
    #     st.info("No database found yet. Upload tables in the Admin panel to create 'uploaded_db.sqlite'.")
    #     # Still allow user to type prompt, but we can't run without DB.
    # conn = None
    # if os.path.exists(db_path):
    #     conn = sqlite3.connect(db_path)
    
    if use_cloud:
        conn = sqlitecloud.connect(
            "sqlitecloud://cbwb6jhxhk.g1.sqlite.cloud:8860/user_info?apikey=tzKSY69TJgit4JxRZqGYxSSSXXn5EWfmoYezjolRdn8"
        )
    else:
        conn = sqlite3.connect("uploaded_db.sqlite")

    # History view
    if show_history:
        st.subheader("🕒 Query History")
        if st.session_state.history:
            for i, rec in enumerate(reversed(st.session_state.history), 1):
                with st.expander(f"Query {i}: {rec['nl']}"):
                    st.markdown("**SQL**")
                    st.code(rec["sql"], language="sql")
                    if isinstance(rec["result"], pd.DataFrame):
                        st.markdown("**Result**")
                        st.dataframe(rec["result"], use_container_width=True, height=300)
                    else:
                        st.markdown("**Result**")
                        st.write(rec["result"])
        else:
            st.info("No queries run yet.")
        if conn:
            conn.close()
        return

    # Main panel
    st.subheader("Ask the question")
    nl_request = st.text_area("", placeholder="e.g., Calculate total number of treated patients by region whose age > 18")

    # Build schema (if DB available)
    schema_text = "No schema available (database not found)."
    schema_summary = ""
    if conn:
        schema_text, schema_map = load_schema(conn)
        schema_summary = summarize_schema(schema_map)
        with st.expander("📚 Database schema"):
            st.text(schema_text)
        # with st.expander("📝 Schema summary"):
        #     st.text(schema_summary)

    # Generate SQL
    if st.button("🧠 Generate SQL"):
        if not nl_request.strip():
            st.warning("Please enter a request first.")
        elif not conn:
            st.error("No database file found. Please create/upload tables in the Admin panel first.")
        else:
            gen_prompt = build_generation_prompt(nl_request, schema_text, schema_summary)
            sql = call_llm_sql(gen_prompt, temperature=temperature, model=model_name)

            ok, msg = enforce_select_only(sql)
            if not ok:
                st.error(f"Guardrail: {msg}")
            else:
                st.markdown("**Proposed SQL**")
                st.session_state["proposed_sql"] = sql
                st.code(sql, language="sql")

                tables_ref = extract_tables_from_sql(sql)
                if tables_ref:
                    st.caption("Tables referenced: " + ", ".join(tables_ref))

    # Allow editing/confirmation of the proposed SQL
    if "proposed_sql" in st.session_state:
        st.markdown("### Review & Run")
        edited_sql = st.text_area("Edit Query (optional)", value=st.session_state["proposed_sql"], height=160, key="editable_sql")

        col_run, col_clear = st.columns([1, 1])
        with col_run:
            if st.button("▶️ Run SQL"):
                if not conn:
                    st.error("No database found.")
                else:
                    ok, msg = enforce_select_only(edited_sql or "")
                    if not ok:
                        st.error(f"Guardrail: {msg}")
                    else:
                        df, err = safe_run_sql(conn, edited_sql)
                        if err:
                            st.error(f"Execution error: {err}")
                            # Try a single self-correction round
                            st.info("Attempting to auto-correct the query…")
                            fix_prompt = build_correction_prompt(edited_sql, err, schema_text, nl_request, schema_summary)
                            fixed_sql = call_llm_sql(fix_prompt, temperature=0.0, model=model_name)

                            ok2, msg2 = enforce_select_only(fixed_sql)
                            if not ok2:
                                st.error(f"Correction failed guardrail: {msg2}")
                            else:
                                st.markdown("**Corrected SQL**")
                                st.code(fixed_sql, language="sql")
                                df2, err2 = safe_run_sql(conn, fixed_sql)
                                if err2:
                                    st.error(f"Correction failed: {err2}")
                                else:
                                    st.success("Corrected query executed successfully.")
                                    st.dataframe(df2, use_container_width=True)
                                    # Save history
                                    st.session_state.history.append({
                                        "nl": nl_request,
                                        "sql": fixed_sql,
                                        "result": df2
                                    })
                                    # Offer download
                                    csv = df2.to_csv(index=False).encode("utf-8")
                                    st.download_button("Download CSV", csv, "query_results.csv", "text/csv")
                                    # Update proposed sql
                                    st.session_state["proposed_sql"] = fixed_sql
                        else:
                            st.success("Query executed successfully.")
                            st.dataframe(df, use_container_width=True)
                            # Save history
                            st.session_state.history.append({
                                "nl": nl_request,
                                "sql": edited_sql,
                                "result": df
                            })
                            # Offer download
                            csv = df.to_csv(index=False).encode("utf-8")
                            st.download_button("Download CSV", csv, "query_results.csv", "text/csv")

        with col_clear:
            if st.button("🧹 Clear Proposed SQL"):
                st.session_state.pop("proposed_sql", None)
                st.rerun()

    if conn:
        conn.close()


























































# import os
# import re
# import sqlite3
# import pandas as pd
# import streamlit as st
# from dotenv import load_dotenv
# from openai import AzureOpenAI
# import sqlitecloud  # Use sqlitecloud

# # Load .env file
# load_dotenv()

# # -----------------------------
# # Azure OpenAI client
# # -----------------------------
# client = AzureOpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_version=os.getenv("OPENAI_API_VERSION"),
# )

# # -----------------------------
# # Helpers
# # -----------------------------
# FORBIDDEN_SQL = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|ATTACH|REINDEX|VACUUM)\b", re.IGNORECASE)


# def get_user_tables(conn) -> list:
#     """Return list of user table names, excluding internal sqlite and _sqliteai_* tables.
#     Works for both local sqlite and sqlitecloud connections.
#     """
#     cursor = conn.cursor()

#     # Try PRAGMA table_list first (works on sqlitecloud and newer SQLite)
#     try:
#         cursor.execute("PRAGMA table_list;")
#         rows = cursor.fetchall()  # Get all tables name and other metadata
#         if rows:
#             # rows format: (schema, name, type, ncol, wr, strict)
#             tables = [r[1] for r in rows if len(r) >= 3 and r[2] == "table"]
#         else:
#             tables = []
#     except Exception:
#         tables = []

#     # If PRAGMA didn't return results, fallback to sqlite_master
#     if not tables:
#         try:
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#             rows = cursor.fetchall()
#             tables = [r[0] for r in rows] if rows else []
#         except Exception:
#             tables = []

#     # Filter out internal/system tables
#     user_tables = [t for t in tables if t and not t.startswith("sqlite_") and not t.startswith("_sqliteai_")]
#     return user_tables


# def load_schema(conn: sqlite3.Connection) -> tuple[str, dict]:
#     """
#     Returns:
#       schema_str: human-readable schema text for prompting
#       schema_map: dict {table_name: [(col_name, col_type), ...]}
#     """
#     tables = get_user_tables(conn)

#     schema_parts = []
#     schema_map = {}

#     for table in tables:
#         # PRAGMA table_info requires the table name quoted if it has unusual chars
#         try:
#             df_info = pd.read_sql(f"PRAGMA table_info('{table}');", conn)
#         except Exception:
#             # try without quotes as a last resort
#             df_info = pd.read_sql(f"PRAGMA table_info({table});", conn)

#         cols = [(r["name"], r["type"]) for _, r in df_info.iterrows()]
#         schema_map[table] = cols
#         cols_text = ", ".join([f"{c} ({t or 'TEXT'})" for c, t in cols])
#         schema_parts.append(f"Table: {table}\nColumns: {cols_text}")

#     schema_str = "\n\n".join(schema_parts) if schema_parts else "No user tables found."
#     return schema_str, schema_map


# def summarize_schema(schema_map: dict) -> str:
#     """
#     Lightweight NL summary (no extra LLM call).
#     """
#     lines = []
#     for t, cols in schema_map.items():
#         colnames = [c for c, _ in cols]
#         key_hints = [c for c in colnames if c.lower().endswith("_id") or c.lower() in {"id", "patient_id", "physician_id"}]
#         lines.append(
#             f"- {t}: {len(cols)} columns; possible keys: {', '.join(key_hints) if key_hints else 'n/a'}."
#         )
#     return "\n".join(lines)


# FEW_SHOT = """
#         Examples (SQLite):

#         User: Count unique patients by region where age > 18
#         SQL:
#         SELECT pd.Region, COUNT(DISTINCT p.Patient_ID) AS treated_patients
#         FROM SQL_103_Assessment_Set1_Data_Claims_Data c
#         JOIN SQL_103_Assessment_Set1_Data_Patient_Demographics p ON p.Patient_ID = c.Patient_ID
#         JOIN SQL_103_Assessment_Set1_Data_Physician_Demographics pd ON pd.Physician_ID = c.Physician_ID
#         WHERE p.Age > 18
#         GROUP BY pd.Region;

#         User: List 10 most recent claims with physician specialty
#         SQL:
#         SELECT c.Claim_ID, c.Date, c.Patient_ID, d.Specialty
#         FROM SQL_103_Assessment_Set1_Data_Claims_Data c
#         JOIN SQL_103_Assessment_Set1_Data_Physician_Demographics d
#           ON d.Physician_ID = c.Physician_ID
#         ORDER BY c.Date DESC
#         LIMIT 10;
#         """


# def build_generation_prompt(nl_request: str, schema_text: str, schema_summary: str) -> str:
#     return f"""
#         You are an expert SQL assistant. Output ONLY a valid SQLite SELECT statement, nothing else.
#         Constraints:
#         - Only SELECT queries are allowed.
#         - Use JOINs when needed.
#         - Use explicit table names from the schema.
#         - If grouping/aggregation is required, ensure proper GROUP BY.
#         - Do not include backticks or markdown fences.
#         - Avoid non-SQL commentary.
#         - Make conservative assumptions if names are ambiguous.

#         Database schema:
#         {schema_text}

#         Schema summary:
#         {schema_summary}

#         {FEW_SHOT}

#         Natural language request:
#         {nl_request}

#         Return only the SQL SELECT query.
#         """


# def build_correction_prompt(original_sql: str, error_msg: str, schema_text: str, nl_request: str, schema_summary: str) -> str:
#     return f"""
#         The following SQLite SELECT query failed. Fix it and return ONLY a corrected SELECT query.

#         Original request:
#         {nl_request}

#         Schema:
#         {schema_text}

#         Schema summary:
#         {schema_summary}

#         Original SQL:
#         {original_sql}

#         Error:
#         {error_msg}

#         Rules:
#         - Return only a valid SQLite SELECT query (no DDL/DML).
#         - Use tables/columns that exist.
#         - No markdown code fences.
#         """


# def call_llm_sql(prompt: str, temperature: float = 0.0, model: str = "gpt-4o-mini") -> str:
#     resp = client.chat.completions.create(
#         model=model,
#         temperature=temperature,
#         messages=[
#             {"role": "system", "content": "You generate safe, strictly-SELECT SQLite queries only."},
#             {"role": "user", "content": prompt},
#         ],
#     )
#     sql = resp.choices[0].message.content.strip()
#     # Strip code fences if the model adds them
#     sql = sql.replace("```sql", "").replace("```", "").strip()
#     return sql


# def enforce_select_only(sql: str) -> tuple[bool, str]:
#     """
#     Returns (ok, message). ok=False means block execution.
#     """
#     if not sql.strip().lower().startswith("select"):
#         return False, "Only SELECT queries are allowed."
#     if FORBIDDEN_SQL.search(sql):
#         return False, "Detected a non-SELECT operation. Only SELECT is allowed."
#     return True, ""


# def extract_tables_from_sql(sql: str) -> list[str]:
#     """
#     Very simple heuristic to extract table names after FROM / JOIN.
#     """
#     candidates = []
#     # get tokens following FROM or JOIN up to whitespace or punctuation
#     for kw in [" from ", "\nfrom ", " join ", "\njoin "]:
#         parts = re.split(kw, sql, flags=re.IGNORECASE)
#         if len(parts) > 1:
#             for p in parts[1:]:
#                 token = re.split(r"[\s\(\),;]+", p.strip())[0]
#                 if token and token.upper() not in {"SELECT"}:
#                     candidates.append(token)
#     # Deduplicate preserving order
#     seen = set()
#     ordered = []
#     for t in candidates:
#         if t not in seen:
#             seen.add(t)
#             ordered.append(t)
#     return ordered


# def safe_run_sql(conn: sqlite3.Connection, sql: str) -> tuple[pd.DataFrame | None, str | None]:
#     try:
#         df = pd.read_sql(sql, conn)
#         return df, None
#     except Exception as e:
#         return None, str(e)


# # -----------------------------
# # Streamlit UI
# # -----------------------------
# def user_panel(use_cloud: bool = True):
#     st.title("📊 SQL Query Assistant")

#     # Session state
#     if "history" not in st.session_state:
#         st.session_state.history = []
#     if "show_history" not in st.session_state:
#         st.session_state.show_history = False

#     # Sidebar
#     show_history = st.sidebar.toggle(
#         "Query History",
#         # value=st.session_state.show_history,
#         value=st.session_state.get("show_history", False),
#         key="show_history",
#         help="Toggle to view previous queries & results",
#     )
#     temperature = st.sidebar.slider("Model temperature", 0.0, 1.0, 0.0, 0.1)
#     model_name = "gpt-4o-mini"

#     # Database connection
#     conn = None
#     try:
#         if use_cloud:
#             conn = sqlitecloud.connect(
#                 "sqlitecloud://cbwb6jhxhk.g1.sqlite.cloud:8860/user_info?apikey=tzKSY69TJgit4JxRZqGYxSSSXXn5EWfmoYezjolRdn8"
#             )
#         else:
#             if os.path.exists("uploaded_db.sqlite"):
#                 conn = sqlite3.connect("uploaded_db.sqlite")
#     except Exception as e:
#         st.error(f"DB connection error: {e}")
#         conn = None

#     # History view
#     if show_history:
#         st.subheader("🕒 Query History")
#         if st.session_state.history:
#             for i, rec in enumerate(reversed(st.session_state.history), 1):
#                 with st.expander(f"Query {i}: {rec['nl']}"):
#                     st.markdown("**SQL**")
#                     st.code(rec["sql"], language="sql")
#                     if isinstance(rec["result"], pd.DataFrame):
#                         st.markdown("**Result**")
#                         st.dataframe(rec["result"], use_container_width=True, height=300)
#                     else:
#                         st.markdown("**Result**")
#                         st.write(rec["result"])
#         else:
#             st.info("No queries run yet.")
#         if conn:
#             conn.close()
#         return

#     # Main panel
#     # st.subheader("Ask the Question")
#     nl_request = st.text_area("Enter your query here:", placeholder="e.g., Calculate total number of treated patients by region whose age > 18")

#     # Build schema (if DB available)
#     schema_text = "No schema available (database not found)."
#     schema_summary = ""
#     if conn:
#         schema_text, schema_map = load_schema(conn)
#         schema_summary = summarize_schema(schema_map)
#         with st.expander("📚 Database schema"):
#             st.text(schema_text)

#     # Generate SQL
#     if st.button("🧠 Generate SQL"):
#         if not nl_request.strip():
#             st.warning("Please enter a request first.")
#         elif not conn:
#             st.error("No database file found. Please create/upload tables in the Admin panel first.")
#         else:
#             gen_prompt = build_generation_prompt(nl_request, schema_text, schema_summary)
#             sql = call_llm_sql(gen_prompt, temperature=temperature, model=model_name)

#             ok, msg = enforce_select_only(sql)
#             if not ok:
#                 st.error(f"Guardrail: {msg}")
#             else:
#                 st.markdown("**Proposed SQL**")
#                 st.session_state["proposed_sql"] = sql
#                 st.code(sql, language="sql")

#                 tables_ref = extract_tables_from_sql(sql)
#                 if tables_ref:
#                     st.caption("Tables referenced: " + ", ".join(tables_ref))

#     # Allow editing/confirmation of the proposed SQL
#     if "proposed_sql" in st.session_state:
#         st.markdown("### Review & Run")
#         edited_sql = st.text_area("Edit Query (optional)", value=st.session_state["proposed_sql"], height=160, key="editable_sql")

#         col_run, col_clear = st.columns([1, 1])
#         with col_run:
#             if st.button("▶️ Run SQL"):
#                 if not conn:
#                     st.error("No database found.")
#                 else:
#                     ok, msg = enforce_select_only(edited_sql or "")
#                     if not ok:
#                         st.error(f"Guardrail: {msg}")
#                     else:
#                         df, err = safe_run_sql(conn, edited_sql)
#                         if err:
#                             st.error(f"Execution error: {err}")
#                             # Try a single self-correction round
#                             st.info("Attempting to auto-correct the query…")
#                             fix_prompt = build_correction_prompt(edited_sql, err, schema_text, nl_request, schema_summary)
#                             fixed_sql = call_llm_sql(fix_prompt, temperature=0.0, model=model_name)

#                             ok2, msg2 = enforce_select_only(fixed_sql)
#                             if not ok2:
#                                 st.error(f"Correction failed guardrail: {msg2}")
#                             else:
#                                 st.markdown("**Corrected SQL**")
#                                 st.code(fixed_sql, language="sql")
#                                 df2, err2 = safe_run_sql(conn, fixed_sql)
#                                 if err2:
#                                     st.error(f"Correction failed: {err2}")
#                                 else:
#                                     st.success("Corrected query executed successfully.")
#                                     st.dataframe(df2, use_container_width=True)
#                                     # Save history
#                                     st.session_state.history.append({
#                                         "nl": nl_request,
#                                         "sql": fixed_sql,
#                                         "result": df2
#                                     })
#                                     # Offer download
#                                     csv = df2.to_csv(index=False).encode("utf-8")
#                                     st.download_button("Download CSV", csv, "query_results.csv", "text/csv")
#                                     # Update proposed sql
#                                     st.session_state["proposed_sql"] = fixed_sql
#                         else:
#                             st.success("Query executed successfully.")
#                             st.dataframe(df, use_container_width=True)
#                             # Save history
#                             st.session_state.history.append({
#                                 "nl": nl_request,
#                                 "sql": edited_sql,
#                                 "result": df
#                             })
#                             # Offer download
#                             csv = df.to_csv(index=False).encode("utf-8")
#                             st.download_button("Download CSV", csv, "query_results.csv", "text/csv")

#         with col_clear:
#             if st.button("🧹 Clear Proposed SQL"):
#                 st.session_state.pop("proposed_sql", None)
#                 st.rerun()

#     if conn:
#         conn.close()




















































































































