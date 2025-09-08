from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Any
from sqlglot import parse_one
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import pandas as pd

from step_1 import extract_schema, generate_embedding_text

from sql_txt_2 import (
    generate_embeddings_hf,
    build_faiss_vectorstore,
    build_chroma_vectorstore,
    build_schema_kg,
)

from sql_text_3 import (
    llm_complete,
    run_sql_query,
)

from sql_text_4 import (
    get_table_names_from_kg,
    validate_sql_with_kg,
)

# -------- Load env --------
load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# ---------- Define State ----------
class PipelineState(TypedDict):
    nl_query: str
    vectorstore: Any
    kg: Any
    semantic_context: str
    rels: list[str]
    canonical_tables: list[str]
    sql_query: str
    df: Any
    validation_hints: str
    is_valid: bool
    retries: int
    visualize: bool
    visualization_path: str

# ------------------------- Nodes ------------------
def retriever_node(state: PipelineState):
    retriever = state["vectorstore"].as_retriever(search_kwargs = {"k":15})
    retrieved_docs = retriever.invoke(state["nl_query"])
    semantic_context = "\n".join([doc.page_content for doc in retrieved_docs])
    state["semantic_context"] = semantic_context
    return state

def kg_node(state: PipelineState):
    table_map = get_table_names_from_kg(state["kg"])
    canonical_tables = list(table_map.values())
    rels = []

    for u, v, data in state["kg"].edges(data = True):
        if isinstance(u,tuple) and isinstance(v,tuple) and u[0] == "table" and v[0] == "table":
            rels.append(f"{u[1]} -> {v[1]} ({data.get('relation')})")
    
    state["canonical_tables"] = canonical_tables
    state["rels"] = rels
    return state

def llm_sql_node(state: PipelineState):
    system_prompt = (
        f"You are an expert SQL generator for SQLite.\n"
        "Do not use any alias in SQL Query\n"
        f"Use ONLY these exact table names (do NOT singularize or invent names): {state["canonical_tables"]}\n"
        "Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE.\n"
        "Take care of values in the table irrespective user asked in capital or in lower case.\n"
        "Return ONLY the final SELECT SQL (no explanation, no backticks).\n"
        "Use provided schema context and join hints. Ensure the SQL is valid SQLite.\n"
        "You may include subqueries or CTEs using nested JSON objects.\n"
        "Do not use unsupported math functions instead use basic aggregate functions.\n"
        "Do not use nested aggregate functions\n"
    )

    user_prompt = f"""
        NL query:
        {state["nl_query"]}

        Schema context:
        {state["semantic_context"]}

        Known relationships:
        {state["rels"]}

        Validation issues:
        {state.get('validation_hints','')}
    """

    sql_query = llm_complete(system_prompt,user_prompt).strip()
    print(f"\n[LLM Raw SQL Attempt #{state['retries'] + 1}] \n{sql_query}")
    state["sql_query"] = sql_query
    return state

def validator_node(state: PipelineState):
    sql_query = state["sql_query"]
    try:
        parse_one(sql_query,dialect = "sqlite")
    except Exception as e:
        print(f"\n❌ sqlglot parsing failed: {e}")
        state["is_valid"] = False
        state["validation_hints"] = f"Parsing error: {e}"
        return state
    
    is_valid, hints = validate_sql_with_kg(sql_query,state["kg"])
    state["is_valid"] = is_valid
    state["validation_hints"] = hints
    return state
    
def executor_node(state: PipelineState):
    query = state["sql_query"]
    try:
        df = run_sql_query(query)
        state["df"] = df
        state["validation_hints"] = None
        
    except Exception as e:
        state["df"] = None
        state["validation_hints"] = str(e)

    # print(f"\n✅ Final SQL Query:\n{state['sql_query']}")
    # print(f"\n📊 Output:\n{state["df"]}")
    return state

def error_handler_node(state: PipelineState):
    system_prompt = (
        f"You are an expert SQL generator for SQLite.\n"
        "Do not use any alias in SQL Query\n"
        f"Use ONLY these exact table names (do NOT singularize or invent names): {state["canonical_tables"]}\n"
        "Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE.\n"
        "Take care of values in the table irrespective user asked in capital or in lower case.\n"
        "Return ONLY the final SELECT SQL (no explanation, no backticks).\n"
        "Use provided schema context and join hints. Ensure the SQL is valid SQLite.\n"
        "You may include subqueries or CTEs using nested JSON objects.\n"
        "Do not use unsupported math functions instead use basic aggregate functions.\n"
        "Do not use nested aggregate functions\n"
    )

    error_msg = state.get("validation_hints","")
    broken_query = state.get("sql_query","")
    nl_query = state.get("nl_query","")
    semantic_context = state.get("semantic_context", "")
    user_prompt = f"""
        Here is the available context and schema
        {semantic_context}

        You generated this SQL query:
        {broken_query}

        It failed with error:
        {error_msg}

        Original user request:
        {nl_query}

        Please carefully review the schema, the original request, the failed query, and the error.
        Then, rewrite the SQL query so it runs correctly on SQLite.
    """
    fixed_query = llm_complete(system_prompt,user_prompt).strip()
    state["sql_query"] = fixed_query
    state["validation_hints"] = None
    return state

def visualizer_node(state: PipelineState):
    """
    A robust node that inspects the DataFrame and attempts to generate the best
    possible visualization, providing detailed feedback along the way.
    """
    print("\n---ENTERING VISUALIZER NODE---")
    
    df = state.get("df")
    
    # 1. --- Initial Sanity Check ---
    if df is None or df.empty:
        print("DIAGNOSIS: DataFrame is empty or missing. A query that returns no data cannot be visualized.")
        print("ACTION: Skipping visualization.")
        state["visualization_path"] = None
        return state

    print("DIAGNOSIS: DataFrame received. Inspecting its structure...")
    
    # 2. --- Inspect the DataFrame for detailed info ---
    # This is incredibly useful for debugging
    print("DataFrame Info:")
    df.info()
    print("\nDataFrame Head:")
    print(df.head())

    try:
        plt.figure(figsize=(12, 7))
        
        # 3. --- Smarter Plotting Logic ---
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        plot_title = state.get('nl_query', 'Query Result Visualization')
        plot_made = False

        # Case 1: Ideal for bar chart (one category, one number)
        if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
            print("ACTION: Found categorical and numeric data. Creating a bar chart.")
            cat_col, num_col = categorical_cols[0], numeric_cols[0]
            # To avoid clutter, only plot top 15 categories
            top_15 = df.nlargest(15, num_col)
            top_15.plot(kind='bar', x=cat_col, y=num_col, legend=False)
            plt.ylabel(num_col)
            plt.title(f"{num_col} by {cat_col}")
            plt.xticks(rotation=45, ha='right')
            plot_made = True

        # Case 2: Ideal for a line or scatter plot (multiple numeric columns)
        elif len(numeric_cols) >= 2:
            print("ACTION: Found multiple numeric columns. Creating a scatter plot.")
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            df.plot(kind='scatter', x=x_col, y=y_col)
            plt.title(f"{y_col} vs. {x_col}")
            plot_made = True

        # Case 3: Single numeric column (histogram)
        elif len(numeric_cols) == 1:
            print("ACTION: Found a single numeric column. Creating a histogram.")
            df[numeric_cols[0]].plot(kind='hist', bins=20, title=f"Distribution of {numeric_cols[0]}")
            plot_made = True
            
        # Case 4: Single categorical column (value counts)
        elif len(categorical_cols) == 1:
            print("ACTION: Found a single categorical column. Creating a value counts bar chart.")
            col = categorical_cols[0]
            df[col].value_counts().nlargest(20).plot(kind='bar')
            plt.title(f"Top 20 Counts of {col}")
            plt.ylabel("Count")
            plt.xticks(rotation=45, ha='right')
            plot_made = True

        # If no plot was made, we cannot visualize this data structure
        if not plot_made:
            print("DIAGNOSIS: The DataFrame's structure is not suitable for any automatic plotting.")
            state["visualization_path"] = None
            return state

        plt.tight_layout()
        
        # 4. --- Save the file and update state ---
        # Ensure the 'visualizations' directory exists
        output_dir = "visualizations"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filepath = os.path.join(output_dir, "output_visualization.png")
        plt.savefig(filepath)
        plt.close() # Important to free up memory

        print(f"\nSUCCESS: Visualization saved successfully to '{filepath}'")
        state["visualization_path"] = filepath

    except Exception as e:
        # 5. --- Catch-all for any other errors ---
        print(f"\nERROR: An unexpected error occurred during plotting: {e}")
        print("ACTION: Visualization could not be created.")
        state["visualization_path"] = None
    
    return state

# --------------------- Build Graph ----------------------
def build_langgraph(vectorstore,kg,nl_query,max_retries = 2,visualize = True,visualize_data = True):
    graph = StateGraph(PipelineState)

    # Add Nodes
    graph.add_node("retriever",retriever_node)
    graph.add_node("kg",kg_node)
    graph.add_node("llm_sql",llm_sql_node)
    graph.add_node("validator",validator_node)
    graph.add_node("executor",executor_node)
    graph.add_node("error_handler",error_handler_node)
    graph.add_node("visualizer",visualizer_node)

    # Entry Point (important!)
    graph.add_edge(START,"retriever")

    # Flow
    graph.add_edge("retriever","kg")
    graph.add_edge("kg","llm_sql")
    graph.add_edge("llm_sql","validator")

    # Conditional branch after validation
    def validation_router(state: PipelineState):
        if(state["is_valid"]):
            return "executor"
        elif state["retries"] + 1 < max_retries:
            state["retries"] += 1
            return "llm_sql"
        else:
            raise ValueError(
                f"Unable to produce valid SQL after {max_retries} attempts. "
                f"Hints: {state['validation_hints']}"
            )
    
    # This router decides if the executor was successful or not
    def executor_router(state: PipelineState):
        if state.get("validation_hints"):
            # If there was an execution error, go to the error handler
            return "error_handler"
        elif state.get("visualize"):
            # If execution was successful AND visualization is requested, go to visualizer
            return "visualizer"
        else:
            # If execution was successful and no visualization is needed, end
            return "result"
    
    graph.add_conditional_edges("executor",executor_router,{"error_handler":"error_handler","visualizer":"visualizer","result":END})
    graph.add_conditional_edges("validator",validation_router,{"executor":"executor","llm_sql":"llm_sql"})
    
    # Exit Point (important !)
    graph.add_edge("error_handler","executor")
    graph.add_edge("visualizer",END)
    # graph.add_edge("error_handler",END)
    # graph.add_edge("executor",END)

    # Compile
    app = graph.compile()

    # Visualization
    if visualize:
        print(app.get_graph().draw_ascii())
    
    # Initial State
    init_state = {
        "nl_query": nl_query,
        "vectorstore": vectorstore,
        "kg": kg,
        "semantic_context":"",
        "rels":[],
        "canonical_tables": [],
        "sql_query": "",
        "df": None,
        "validation_hints": "",
        "is_valid": False,
        "retries": 0, 
        "visualize": visualize_data,
        "visualization_path": None,
    }

    final_state = app.invoke(init_state)
    return final_state["sql_query"],final_state["df"],final_state.get("visualization_path")

# -------------------------- Example Main -------------------
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
    # nl_query = "Find all the People with region they belong who have returned atleast one order"
    # nl_query = "Orders where the amount is lesser than standard deviation for that Region."
    # nl_query = "List People who never returned an order but belong to Regions where return rate > 30%."
    nl_query = "Find the month with the highest return rate across all orders"
    # nl_query = "For each Region, calculate the percentage contribution of each Person’s spending to the Region’s total."
    # nl_query = "Find the Person_Name who has spent the maximum total Amount"
    sql_df, df, viz_path = build_langgraph(vectorstore,G,nl_query)
    print(f"\n✅ Final SQL Query:\n {sql_df}")
    print(f"\n📊 Output:\n{df}")
    print(f"Visualization Path: {viz_path}")
    if(viz_path):
        print(f"Visualization created at: {viz_path}")