import streamlit as st
import pandas as pd
import re
from graph_logic import create_graph_app
from dotenv import load_dotenv
import os

# Your imports from your custom modules
from step_1 import extract_schema, generate_embedding_text
from sql_txt_2 import (
    generate_embeddings_hf,
    build_faiss_vectorstore,
    build_schema_kg,
)

# --- NEW: Import your database connector directly for the manual query ---
import sqlitecloud

# -------- Load env --------
load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# ===================================================================
# --- NEW: HELPER FUNCTIONS FOR DIRECT SQL EXECUTION ---
# ===================================================================

def enforce_select_only(sql: str):
    """
    A simple but crucial security guardrail. Returns (is_safe, message).
    """
    # Use a case-insensitive search to find forbidden keywords
    forbidden_keywords = re.compile(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|ATTACH|REINDEX|VACUUM)\b",
        re.IGNORECASE
    )
    if not sql.strip().upper().startswith("SELECT"):
        return False, "Query must be a SELECT statement."
    if forbidden_keywords.search(sql):
        return False, "Only SELECT queries are allowed for security reasons."
    return True, ""

def run_direct_sql(connection_string: str, sql_query: str):
    """
    Connects to the database and runs a provided SQL query directly.
    Returns (DataFrame, error_message).
    """
    try:
        with sqlitecloud.connect(connection_string) as conn:
            df = pd.read_sql_query(sql_query, conn)
        return df, None
    except Exception as e:
        return None, str(e)


# ===================================================================
# THE MAIN user_panel() FUNCTION
# ===================================================================
def user_panel():
    # -------------------------------------------------------------------
    # CACHE THE BACKEND
    # -------------------------------------------------------------------
    @st.cache_resource
    def load_backend():
        st.info("Initializing RAG/KG Backend... This may take a moment on first run or after an admin update.")
        schema_info = extract_schema(db_path)
        embedding_docs = generate_embedding_text(schema_info)
        embedded_docs = generate_embeddings_hf(embedding_docs)
        vectorstore = build_faiss_vectorstore(embedded_docs)
        G = build_schema_kg(schema_info)
        app = create_graph_app()
        st.success("Backend initialized successfully!")
        return app, vectorstore, G

    app, vectorstore, kg = load_backend()
    
    # -------------------------------------------------------------------
    # MANAGE SESSION STATE
    # -------------------------------------------------------------------
    if "history" not in st.session_state:
        st.session_state.history = []
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Latest Result"

    # -------------------------------------------------------------------
    # DEFINE THE USER INTERFACE WITH TABS
    # -------------------------------------------------------------------
    st.sidebar.header("Query Options")
    visualize_toggle = st.sidebar.checkbox("Generate Visualization", value=True, help="Create a plot (only for Natural Language queries).")
    
    st.sidebar.divider()
    st.sidebar.header("Display Mode")
    st.session_state.view_mode = st.sidebar.radio(
        "Choose what to display:",
        ["Latest Result", "Full History"],
        label_visibility="collapsed"
    )

    # --- NEW: Create two tabs for the two different functionalities ---
    tab1, tab2 = st.tabs(["**Ask with Natural Language**", "**Run SQL Directly**"])

    # --- TAB 1: Natural Language Query (Your existing workflow) ---
    with tab1:
        with st.form("nl_query_form"):
            user_query = st.text_area("Enter your question in natural language:", height=150, placeholder="e.g., Calculate total number of treated patients by region whose age > 18")
            submit_nl_button = st.form_submit_button("🚀 Process Natural Language Query")
            
            if submit_nl_button and user_query:
                st.session_state.view_mode = "Latest Result"
                with st.spinner("Analyzing query, running through RAG & KG, executing, and self-correcting..."):
                    try:
                        init_state = { "nl_query": user_query, "vectorstore": vectorstore, "kg": kg, "visualize": visualize_toggle, "semantic_context": "", "rels": [], "canonical_tables": [], "sql_query": "", "df": None, "validation_hints": "", "is_valid": False, "retries": 0, "error_retries": 0, "visualization_path": None }
                        final_state = app.invoke(init_state)
                        # Add a 'type' to the history for easier display later
                        st.session_state.history.append({"query": user_query, "result": final_state, "type": "langgraph"})
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {e}")
                        st.session_state.history.append({"query": user_query, "error": str(e), "type": "langgraph"})
    
    # --- TAB 2: Direct SQL Execution (The new feature) ---
    with tab2:
        with st.form("sql_query_form"):
            manual_sql_query = st.text_area("Enter your SQLite query:", height=200, placeholder='SELECT "Region", COUNT(*)\nFROM "Patients"\nGROUP BY "Region";')
            submit_sql_button = st.form_submit_button("▶️ Run Manual SQL")

            if submit_sql_button and manual_sql_query:
                st.session_state.view_mode = "Latest Result"
                is_safe, message = enforce_select_only(manual_sql_query)
                if not is_safe:
                    st.error(f"Security Alert: {message}")
                else:
                    with st.spinner("Executing your SQL query..."):
                        df, error = run_direct_sql(db_path, manual_sql_query)
                        if error:
                            st.error(f"Execution Failed: {error}")
                            st.session_state.history.append({"query": "Manual SQL Execution", "sql": manual_sql_query, "error": error, "type": "manual"})
                        else:
                            st.success("Manual query executed successfully!")
                            st.session_state.history.append({"query": "Manual SQL Execution", "sql": manual_sql_query, "df": df, "type": "manual"})

    # -------------------------------------------------------------------
    # --- UPDATED: CONDITIONAL DISPLAY LOGIC TO HANDLE BOTH RESULT TYPES ---
    # -------------------------------------------------------------------
    st.divider()

    # --- VIEW 1: Show the most recent result ---
    if st.session_state.view_mode == "Latest Result":
        if st.session_state.history:
            latest_entry = st.session_state.history[-1]
            st.subheader(f"Latest Result for: \"{latest_entry['query']}\"")
            
            if "error" in latest_entry:
                st.error(f"Failed to process query: {latest_entry['error']}")
            # Display logic for LangGraph results
            elif latest_entry.get("type") == "langgraph":
                final_state = latest_entry["result"]
                st.code(final_state.get("sql_query"), language="sql")
                st.dataframe(final_state.get("df"))
                if final_state.get("visualization_path"):
                    st.image(final_state.get("visualization_path"))
            # Display logic for Manual SQL results
            elif latest_entry.get("type") == "manual":
                st.code(latest_entry.get("sql"), language="sql")
                st.dataframe(latest_entry.get("df"))
        else:
            st.info("Submit a query to see the results here.")

    # --- VIEW 2: Show the entire history in detail ---
    elif st.session_state.view_mode == "Full History":
        st.subheader("📜 Full Query History")
        if not st.session_state.history:
            st.info("No queries have been run in this session yet.")
        else:
            for i, entry in enumerate(reversed(st.session_state.history)):
                with st.expander(f"Query {len(st.session_state.history) - i}: {entry['query']}", expanded=(i==0)):
                    if "error" in entry:
                        st.error(f"Failed to process query: {entry['error']}")
                    # Display for LangGraph history
                    elif entry.get("type") == "langgraph":
                        final_state = entry["result"]
                        st.markdown("##### Final SQL Query"); st.code(final_state.get("sql_query"), language="sql")
                        st.markdown("##### Query Result Data"); st.dataframe(final_state.get("df"))
                        if final_state.get("visualization_path"):
                            st.markdown("##### Data Visualization"); st.image(final_state.get("visualization_path"))
                    # Display for Manual SQL history
                    elif entry.get("type") == "manual":
                        st.markdown("##### Manual SQL Query"); st.code(entry.get("sql"), language="sql")
                        st.markdown("##### Query Result Data"); st.dataframe(entry.get("df"))
























# import streamlit as st
# import pandas as pd
# from graph_logic import create_graph_app
# from dotenv import load_dotenv
# import os

# # Your imports from your custom modules
# from step_1 import extract_schema, generate_embedding_text
# from sql_txt_2 import (
#     generate_embeddings_hf,
#     build_faiss_vectorstore,
#     build_schema_kg,
# )
# # Note: The functions from sql_text_3 and sql_text_4 are now inside your graph_logic.py nodes
# # so they are no longer needed here.

# # -------- Load env --------
# load_dotenv()
# db_path = os.getenv("DATABASE_CONNECTION_STRING")

# # ===================================================================
# # THE ENTIRE PAGE LOGIC IS WRAPPED IN THIS user_panel() FUNCTION
# # ===================================================================
# def user_panel():
#     # -------------------------------------------------------------------
#     # CACHE THE BACKEND
#     # -------------------------------------------------------------------
#     @st.cache_resource
#     def load_backend():
#         """
#         Initializes all backend components: Vectorstore, KG, and the compiled
#         LangGraph app.
#         """
#         st.info("Initializing RAG/KG Backend... This may take a moment on first run or after an admin update.")
        
#         # This logic for building the RAG/KG system is now part of the cached setup.
#         # It will only run when the app starts or when the cache is cleared.
#         schema_info = extract_schema(db_path)
#         embedding_docs = generate_embedding_text(schema_info)
#         embedded_docs = generate_embeddings_hf(embedding_docs)
#         vectorstore = build_faiss_vectorstore(embedded_docs)
#         G = build_schema_kg(schema_info)
        
#         app = create_graph_app()
        
#         st.success("Backend initialized successfully!")
#         return app, vectorstore, G

#     app, vectorstore, kg = load_backend()
    
#     # -------------------------------------------------------------------
#     # MANAGE SESSION STATE
#     # -------------------------------------------------------------------
#     if "history" not in st.session_state:
#         st.session_state.history = []
    
#     # --- NEW: Add a session state for the view mode ---
#     if "view_mode" not in st.session_state:
#         st.session_state.view_mode = "Latest Result"

#     # -------------------------------------------------------------------
#     # DEFINE THE USER INTERFACE
#     # -------------------------------------------------------------------
#     st.sidebar.header("Query Options")
#     visualize_toggle = st.sidebar.checkbox("Generate Visualization", value=True, help="Create a plot of the query results.")
    
#     # --- NEW: Add a toggle to switch between views ---
#     st.sidebar.divider()
#     st.sidebar.header("Display Mode")
#     st.session_state.view_mode = st.sidebar.radio(
#         "Choose what to display in the main panel:",
#         ["Latest Result", "Full History"],
#         label_visibility="collapsed" # Hides the "Choose what..." label
#     )

#     with st.form("query_form"):
#         user_query = st.text_area("Enter your question in natural language:", height=150, placeholder="e.g., Calculate total number of treated patients by region whose age > 18")
#         submit_button = st.form_submit_button("🚀 Process Query")
    
#     # -------------------------------------------------------------------
#     # GRAPH INVOCATION LOGIC
#     # -------------------------------------------------------------------
#     if submit_button and user_query:
#         # When a new query is submitted, always switch the view to show the latest result
#         st.session_state.view_mode = "Latest Result"
        
#         with st.spinner("Analyzing query, running through RAG & KG, executing, and self-correcting..."):
#             try:
#                 init_state = {
#                     "nl_query": user_query,
#                     "vectorstore": vectorstore,
#                     "kg": kg,
#                     "visualize": visualize_toggle,
#                     "semantic_context": "", "rels": [], "canonical_tables": [],
#                     "sql_query": "", "df": None, "validation_hints": "",
#                     "is_valid": False, "retries": 0, "error_retries": 0,
#                     "visualization_path": None,
#                 }
#                 final_state = app.invoke(init_state)
#                 st.session_state.history.append({"query": user_query, "result": final_state})
#             except Exception as e:
#                 st.error(f"An unexpected error occurred: {e}")
#                 st.session_state.history.append({"query": user_query, "error": str(e)})

#     # -------------------------------------------------------------------
#     # --- NEW: CONDITIONAL DISPLAY LOGIC BASED ON VIEW MODE ---
#     # -------------------------------------------------------------------
#     st.divider()

#     # --- VIEW 1: Show only the most recent result ---
#     if st.session_state.view_mode == "Latest Result":
#         if st.session_state.history:
#             latest_entry = st.session_state.history[-1]
#             st.subheader(f"Latest Result for: \"{latest_entry['query']}\"")
            
#             if "error" in latest_entry:
#                 st.error(f"Failed to process query: {latest_entry['error']}")
#             else:
#                 # Unpack and display the full result
#                 final_state = latest_entry["result"]
#                 sql_query = final_state.get("sql_query")
#                 df = final_state.get("df")
#                 viz_path = final_state.get("visualization_path")

#                 with st.expander("Final SQL Query", expanded=False):
#                     st.code(sql_query, language="sql")
                
#                 st.subheader("Query Result Data")
#                 if df is not None and not df.empty:
#                     st.dataframe(df)
#                     csv = df.to_csv(index=False).encode("utf-8")
#                     st.download_button("📥 Download Results as CSV", csv, "query_results.csv", "text/csv")
#                 else:
#                     st.warning("The query executed successfully but returned no data.")

#                 if viz_path:
#                     st.subheader("📊 Data Visualization")
#                     st.image(viz_path, caption="A plot generated from the query results.")
#         else:
#             st.info("Submit a query to see the results here.")

#     # --- VIEW 2: Show the entire history in detail ---
#     elif st.session_state.view_mode == "Full History":
#         st.subheader("📜 Full Query History")
#         if not st.session_state.history:
#             st.info("No queries have been run in this session yet.")
#         else:
#             # Iterate through history in reverse (most recent first)
#             for i, entry in enumerate(reversed(st.session_state.history)):
#                 with st.expander(f"Query #{len(st.session_state.history) - i}: {entry['query']}", expanded=(i==0)):
#                     if "error" in entry:
#                         st.error(f"Failed to process query: {entry['error']}")
#                     else:
#                         # Unpack and display the full result for this specific history item
#                         final_state = entry["result"]
#                         sql_query = final_state.get("sql_query")
#                         df = final_state.get("df")
#                         viz_path = final_state.get("visualization_path")

#                         st.markdown("##### Generated SQL Query")
#                         st.code(sql_query, language="sql")
                        
#                         st.markdown("##### Query Result Data")
#                         if df is not None and not df.empty:
#                             st.dataframe(df)
#                             csv = df.to_csv(index=False).encode("utf-8")
#                             # Use a unique key for each download button
#                             st.download_button("📥 Download as CSV", csv, f"query_{i}_results.csv", "text/csv", key=f"download_csv_{i}")
#                         else:
#                             st.warning("Query returned no data.")

#                         if viz_path:
#                             st.markdown("##### Data Visualization")
#                             st.image(viz_path)























# import streamlit as st
# import pandas as pd
# from graph_logic import create_graph_app
# from dotenv import load_dotenv
# import os
# from step_1 import extract_schema, generate_embedding_text
# from sql_txt_2 import (
#     generate_embeddings_hf,
#     build_faiss_vectorstore,
#     build_schema_kg,
# )
# from sql_text_3 import (
#     llm_complete,
#     run_sql_query,
# )
# from sql_text_4 import (
#     get_table_names_from_kg,
#     validate_sql_with_kg,
# )

# # -------- Load env --------
# load_dotenv()
# db_path = os.getenv("DATABASE_CONNECTION_STRING")

# st.set_page_config(page_title="Query Assistant", layout="wide")
# st.title("🤖 RAG + KG Powered Query Assistant")

# # ===================================================================
# # THE ENTIRE PAGE LOGIC IS NOW WRAPPED IN THIS user_panel() FUNCTION
# # ===================================================================
# def user_panel():
#     # -------------------------------------------------------------------
#     # STEP 1: CACHE THE BACKEND (MOST IMPORTANT STEP)
#     # This function initializes all heavy components and is cached. It will only
#     # run once when the app starts, or when the cache is cleared from the Admin Page.
#     # -------------------------------------------------------------------
#     @st.cache_resource
#     def load_backend():
#         """
#         Initializes all backend components: Vectorstore, KG, and the compiled
#         LangGraph app.
#         """
#         st.info("Initializing backend... This may take a moment on first run.")
#         # Extract schema
#         schema_info = extract_schema(db_path)
#         # Embeddings
#         embedding_docs = generate_embedding_text(schema_info)
#         embedded_docs = generate_embeddings_hf(embedding_docs)
#         # Vectorstore (FAISS or Chroma)
#         vectorstore = build_faiss_vectorstore(embedded_docs)
#         # Build Knowledge Graph
#         G = build_schema_kg(schema_info)
#         # Compile the LangGraph application
#         app = create_graph_app()

#         st.success("Backend initialized successfully!")
#         return app, vectorstore, G

#     # Load the cached backend components. Streamlit will not re-run load_backend()
#     # on subsequent interactions unless the cache is cleared.
#     app, vectorstore, kg = load_backend()

#     # -------------------------------------------------------------------
#     # STEP 2: MANAGE SESSION STATE FOR HISTORY
#     # We will now store the entire final state of the graph for richer history.
#     # -------------------------------------------------------------------
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     # -------------------------------------------------------------------
#     # STEP 3: DEFINE THE USER INTERFACE
#     # The UI is simplified. We have one main action button.
#     # -------------------------------------------------------------------
#     st.sidebar.header("Query Options")
#     visualize_toggle = st.sidebar.checkbox("Generate Visualization", value=True, help="If checked, the system will attempt to create a plot of the query results.")

#     with st.form("query_form"):
#         user_query = st.text_area("Enter your question in natural language:", height=150, placeholder="e.g., Calculate total number of treated patients by region whose age > 18")
#         submit_button = st.form_submit_button("🚀 Process Query")

#     # -------------------------------------------------------------------
#     # STEP 4: REPLACE THE ENTIRE OLD LOGIC WITH A SINGLE GRAPH INVOCATION
#     # -------------------------------------------------------------------
#     if submit_button and user_query:
#         with st.spinner("Analyzing query, running through RAG & KG, executing, and self-correcting..."):
#             try:
#                 # Prepare the initial state dictionary to be passed to the graph
#                 init_state = {
#                     "nl_query": user_query,
#                     "vectorstore": vectorstore,
#                     "kg": kg,
#                     "visualize": visualize_toggle,
#                     # All other keys must be initialized to their default empty values
#                     "semantic_context": "", 
#                     "rels": [], 
#                     "canonical_tables": [],
#                     "sql_query": "", 
#                     "df": None, 
#                     "validation_hints": "",
#                     "is_valid": False, 
#                     "retries": 0, 
#                     "error_retries": 0,
#                     "visualization_path": None,
#                 }

#                 # Invoke the LangGraph application. This one line replaces all your
#                 # previous logic for prompt building, LLM calls, and error correction.
#                 final_state = app.invoke(init_state)

#                 # Store the complete result in the session state history
#                 st.session_state.history.append({"query": user_query, "result": final_state})

#             except Exception as e:
#                 # Catch any hard failures from the graph (e.g., max retries exceeded)
#                 st.error(f"An unexpected error occurred: {e}")
#                 st.session_state.history.append({"query": user_query, "error": str(e)})

#     # -------------------------------------------------------------------
#     # STEP 5: DISPLAY THE FINAL RESULT FROM THE LATEST HISTORY ENTRY
#     # This section is now much simpler. It just unpacks and displays the
#     # final state without needing any further logic.
#     # -------------------------------------------------------------------
#     if st.session_state.history:
#         latest_entry = st.session_state.history[-1]

#         st.divider()
#         st.subheader(f"Results for: \"{latest_entry['query']}\"")

#         if "error" in latest_entry:
#             st.error(f"Failed to process query: {latest_entry['error']}")
#         else:
#             final_state = latest_entry["result"]
#             sql_query = final_state.get("sql_query")
#             df = final_state.get("df")
#             viz_path = final_state.get("visualization_path")

#             # Display the final SQL query
#             with st.expander("Final SQL Query", expanded=False):
#                 st.code(sql_query, language="sql")

#             # Display the result DataFrame
#             st.subheader("Query Result Data")
#             if df is not None and not df.empty:
#                 st.dataframe(df)
#                 # Offer download button
#                 csv = df.to_csv(index=False).encode("utf-8")
#                 st.download_button("📥 Download Results as CSV", csv, "query_results.csv", "text/csv")
#             else:
#                 st.warning("The query executed successfully but returned no data.")

#             # Display the visualization if it was created
#             if viz_path:
#                 st.subheader("📊 Data Visualization")
#                 st.image(viz_path, caption="A plot generated from the query results.")

#     # -------------------------------------------------------------------
#     # STEP 6: DISPLAY HISTORY IN THE SIDEBAR
#     # -------------------------------------------------------------------
#     with st.sidebar.expander("📜 Query History", expanded=True):
#         if not st.session_state.history:
#             st.write("No queries yet.")
#         for i, item in enumerate(reversed(st.session_state.history)):
#             st.info(f"**{len(st.session_state.history) - i}. {item['query']}**")