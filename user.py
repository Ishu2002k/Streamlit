import streamlit as st
import pandas as pd
import re
from graph_logic import create_graph_app
from dotenv import load_dotenv
import os
import sqlitecloud

# Your imports from your custom modules
from step_1 import extract_schema, generate_embedding_text
from sql_txt_2 import (
    generate_embeddings_hf,
    build_faiss_vectorstore,
    build_schema_kg,
)

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

def user_panel():
    # -------------------------------------------------------------------
    # CACHE THE BACKEND
    # -------------------------------------------------------------------
    @st.cache_resource
    def load_backend():
        st.info("Initializing RAG/KG Backend... Please wait.")
        schema_info = extract_schema(db_path)
        embedding_docs = generate_embedding_text(schema_info)
        embedded_docs = generate_embeddings_hf(embedding_docs)
        vectorstore = build_faiss_vectorstore(embedded_docs)
        G = build_schema_kg(schema_info)
        app = create_graph_app()
        st.success("Backend ready ✅")
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
    # SIDEBAR OPTIONS
    # -------------------------------------------------------------------
    st.sidebar.header("Query Options")
    visualize_toggle = st.sidebar.checkbox(
        "Generate Visualization", value=True,
        help="Create a plot (only for Natural Language queries)."
    )
    trace_toggle = st.sidebar.checkbox(
        "Show Execution Trace", value=False,
        help="See intermediate states from LangGraph pipeline."
    )

    st.sidebar.divider()
    st.sidebar.header("Display Mode")
    st.session_state.view_mode = st.sidebar.radio(
        "Choose what to display:",
        ["Latest Result", "Full History"],
        label_visibility="collapsed"
    )

    # -------------------------------------------------------------------
    # TABS: NL Query vs SQL Query
    # -------------------------------------------------------------------
    tab1, tab2 = st.tabs(["**Ask with Natural Language**", "**Run SQL Directly**"])

    # --- TAB 1: Natural Language Query ---
    with tab1:
        with st.form("nl_query_form"):
            user_query = st.text_area(
                "Enter your question in natural language:",
                height=150,
                placeholder="e.g., Calculate total number of treated patients by region whose age > 18"
            )
            submit_nl_button = st.form_submit_button("🚀 Process Query")

# ------------------------------------------------------------------------------------------------------------------------------------------
        # Table Metadata Dropdown (outside the form to avoid form submission issues)
        st.markdown("#### 📊 Database Schema Reference")
        with st.expander("View Table Structure & Columns", expanded=False):
            try:
                # Extract schema information using existing function
                schema_info = extract_schema(db_path)
                
                if schema_info and 'tables' in schema_info:
                    tables_data = schema_info['tables']
                    table_names = list(tables_data.keys())
                    
                    # Create dropdown for table selection
                    selected_table = st.selectbox(
                        "Select a table to view its structure:",
                        options=["-- Select a table --"] + table_names,
                        key="table_selector"
                    )
                    
                    if selected_table and selected_table != "-- Select a table --":
                        table_info = tables_data[selected_table]
                        
                        # Display table information
                        st.markdown(f"**Table: `{selected_table}`**")
                        
                        # Show columns and data types
                        if 'columns' in table_info and table_info['columns']:
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("**Columns:**")
                                for column in table_info['columns']:
                                    col_name = column['name']
                                    st.markdown(f"• `{col_name}`")
                            
                            with col2:
                                st.markdown("**Data Types:**")
                                for column in table_info['columns']:
                                    col_type = column['type']
                                    st.markdown(f"`{col_type}`")
                        
                            # Show sample data examples
                            if 'examples' in table_info and table_info['examples']:
                                st.markdown("**📋 Sample Data (First 3 rows):**")
                                
                                # Create a nice table format for examples
                                import pandas as pd
                                examples_df = pd.DataFrame(table_info['examples'])
                                st.dataframe(examples_df, use_container_width=True)
                    
                    # Show all tables summary
                    st.markdown("---")
                    st.markdown("**📋 All Available Tables:**")
                    
                    for table_name, table_data in tables_data.items():
                        column_count = len(table_data.get('columns', []))
                        example_count = len(table_data.get('examples', []))
                        st.markdown(f"• **{table_name}** ({column_count} columns, {example_count} sample records)")
                    
                else:
                    st.warning("Could not load database schema information or schema is empty.")
                    
            except Exception as e:
                st.error(f"Error loading schema: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.info("Schema information is not available at the moment.")
# ----------------------------------------------------------------------------------------------------------------------------------------

            if submit_nl_button and user_query:
                st.session_state.view_mode = "Latest Result"
                with st.spinner("Analyzing, executing, validating..."):
                    try:
                        init_state = {
                            "nl_query": user_query,
                            "vectorstore": vectorstore,
                            "kg": kg,
                            "visualize": visualize_toggle,
                            "semantic_context": "",
                            "rels": [],
                            "canonical_tables": [],
                            "sql_query": "",
                            "df": None,
                            "validation_hints": "",
                            "is_valid": False,
                            "retries": 0,
                            "error_retries": 0,
                            "trace": []
                        }
                        final_state = app.invoke(init_state)
                        st.session_state.history.append(
                            {"query": user_query, "result": final_state, "type": "langgraph"}
                        )
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
                        st.session_state.history.append(
                            {"query": user_query, "error": str(e), "type": "langgraph"}
                        )

    # --- TAB 2: Manual SQL Query ---
    with tab2:
        with st.form("sql_query_form"):
            manual_sql_query = st.text_area(
                "Enter your SQL query:",
                height=200,
                placeholder='SELECT "Region", COUNT(*) FROM "Patients" GROUP BY "Region";'
            )
            submit_sql_button = st.form_submit_button("▶️ Run SQL")

            if submit_sql_button and manual_sql_query:
                st.session_state.view_mode = "Latest Result"
                is_safe, message = enforce_select_only(manual_sql_query)
                if not is_safe:
                    st.error(f"Security Alert: {message}")
                else:
                    with st.spinner("Executing SQL..."):
                        df, error = run_direct_sql(db_path, manual_sql_query)
                        if error:
                            st.error(f"Execution Failed: {error}")
                            st.session_state.history.append(
                                {"query": "Manual SQL", "sql": manual_sql_query, "error": error, "type": "manual"}
                            )
                        else:
                            st.success("Manual query executed successfully!")
                            st.session_state.history.append(
                                {"query": "Manual SQL", "sql": manual_sql_query, "df": df, "type": "manual"}
                            )

    # -------------------------------------------------------------------
    # RESULTS DISPLAY
    # -------------------------------------------------------------------
    st.divider()

    def show_langgraph_result(final_state, trace=False,unique_key=""):
        st.markdown("##### Final SQL Query")
        st.code(final_state.get("sql_query"), language="sql")

        if final_state.get("df") is not None:
            st.markdown("##### Query Result Data")
            st.dataframe(final_state["df"])
            st.download_button(
                "📥 Download Results as CSV",
                final_state["df"].to_csv(index=False),
                file_name="results.csv",
                mime="text/csv",
                key=f"download_langgraph_{unique_key}"
            )

        if final_state.get("visualization_path"):  # file-based
            st.markdown("##### Data Visualization")
            st.image(final_state["visualization_path"])

        if "retries" in final_state or "error_retries" in final_state:
            st.info(f"Validation retries: {final_state.get('retries',0)} | "
                    f"Execution retries: {final_state.get('error_retries',0)}")

        if final_state.get("validation_hints"):
            st.warning(f"Validator hints: {final_state['validation_hints']}")

        if trace and final_state.get("trace"):
            st.markdown("##### Execution Trace")
            for step in final_state["trace"]:
                with st.expander(f"Step: {step['node']}"):
                    st.json(step)

    # --- Latest Result ---
    if st.session_state.view_mode == "Latest Result":
        if st.session_state.history:
            latest_entry = st.session_state.history[-1]
            st.subheader(f"Latest Result: \"{latest_entry['query']}\"")

            if "error" in latest_entry:
                st.error(f"Failed: {latest_entry['error']}")
            elif latest_entry.get("type") == "langgraph":
                show_langgraph_result(latest_entry["result"], trace=trace_toggle,unique_key="latest")
            elif latest_entry.get("type") == "manual":
                st.code(latest_entry.get("sql"), language="sql")
                st.dataframe(latest_entry.get("df"))
                if latest_entry.get("df") is not None:
                    st.download_button(
                        "📥 Download Results as CSV",
                        latest_entry["df"].to_csv(index=False),
                        file_name="manual_results.csv",
                        mime="text/csv",
                        key="download_manual_latest"
                    )
        else:
            st.info("Submit a query to see results here.")

    # --- Full History ---
    elif st.session_state.view_mode == "Full History":
        st.subheader("📜 Full Query History")
        if not st.session_state.history:
            st.info("No queries have been run yet.")
        else:
            for i, entry in enumerate(reversed(st.session_state.history)):
                history_index = len(st.session_state.history) - i - 1
                with st.expander(f"Query {len(st.session_state.history) - i}: {entry['query']}", expanded=(i == 0)):
                    if "error" in entry:
                        st.error(f"Failed: {entry['error']}")
                    elif entry.get("type") == "langgraph":
                        show_langgraph_result(entry["result"], trace=trace_toggle,unique_key=f"history_{history_index}")
                    elif entry.get("type") == "manual":
                        st.markdown("##### Manual SQL Query")
                        st.code(entry.get("sql"), language="sql")
                        st.markdown("##### Query Result Data")
                        st.dataframe(entry.get("df"))
                        if entry.get("df") is not None:  # Added check for df existence
                            st.download_button(
                                "📥 Download Results as CSV",
                                entry["df"].to_csv(index=False),
                                file_name=f"manual_results_query_{history_index}.csv",
                                mime="text/csv",
                                key=f"download_manual_history_{history_index}"  # Added unique key
                            )