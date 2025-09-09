import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import TypedDict, List, Optional
from openai import AzureOpenAI
from langgraph.graph import StateGraph, START, END
from sqlglot import parse_one
from dotenv import load_dotenv
from sql_text_3 import (
    llm_complete,
    run_sql_query,
)
from sql_text_4 import (
    get_table_names_from_kg,
    validate_sql_with_kg,
)
from fewshot_adapter import (
    build_fewshot_prompt,
    add_fewshot_example,
    save_turn,
    get_conversation_context,
    build_error_correction_prompt,
    add_error_correction_example,
)

# ---------- Azure OpenAI Client ----------
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

LLM_MODEL = os.getenv("LLM_MODEL")
# -------- Load env --------
load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# =================================================================
# 1. DEFINE THE STATE FOR THE GRAPH
# This TypedDict defines the "memory" of your agent. All nodes
# will read from and write to this shared state.
# =================================================================
class PipelineState(TypedDict):
    # Inputs
    nl_query: str                   # The user's natural language question
    vectorstore: any                # Your RAG vectorstore object
    kg: any                         # Your Knowledge Graph object
    visualize: bool                 # Toggle for generating a plot

    # State variables that are populated by nodes
    semantic_context: str           # Context retrieved from the vectorstore
    rels: List[str]                 # Relationships from the Knowledge Graph
    canonical_tables: List[str]     # Table names identified for the query
    sql_query: str                  # The generated SQL query
    df: Optional[pd.DataFrame]      # The DataFrame result of the SQL execution
    
    # Control flow and error handling
    validation_hints: Optional[str] # Error messages from SQL execution or validation
    is_valid: bool                  # Flag from the validator node
    retries: int                    # Counter for the validation loop
    error_retries: int              # New Counter for the execution/correction loop
    
    # Final output
    visualization_path: Optional[str] # Path to the saved visualization image

# =================================================================
# 3. DEFINE THE NODES OF THE GRAPH
# Each node is a Python function that takes the state and returns
# a modified version of the state.
# =================================================================
# --------------------Retriever Nodes ------------------
def retriever_node(state: PipelineState):
    """
    Node 1: Retrieves relevant context using RAG.
    """
    retriever = state["vectorstore"].as_retriever(search_kwargs = {"k":15})
    retrieved_docs = retriever.invoke(state["nl_query"])
    semantic_context = "\n".join([doc.page_content for doc in retrieved_docs])
    state["semantic_context"] = semantic_context
    return state

def kg_node(state: PipelineState):
    """
    Node 2: Enhances context using the Knowledge Graph.
    """
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
    """
    Node 3: Generates the SQL query using the LLM.
    """
    system_prompt = (
        f"You are an expert SQL generator for SQLite.\n"
        # "Do not use any alias in SQL Query\n"
        f"Use ONLY these exact table names (do NOT singularize or invent names): {state["canonical_tables"]}\n"
        "Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE.\n"
        "Take care of values in the table irrespective user asked in capital or in lower case.\n"
        "Return ONLY the final SELECT SQL (no explanation, no backticks).\n"
        "Use provided schema context and join hints. Ensure the SQL is valid SQLite.\n"
        "You may include subqueries or CTEs using nested JSON objects.\n"
        "Do not use unsupported math functions instead use basic aggregate functions.\n"
        "Do not use nested aggregate functions\n"
    )

    # ---- Conversation context -------
    conv_context = get_conversation_context(max_turns = 3)

    # ------- Few-shot dynamic prompt ------
    fewshot_prompt = build_fewshot_prompt(state["nl_query"])

    # ------- User prompt with schema + validation -------
    user_prompt = f"""
        {conv_context}

        {fewshot_prompt}

        NL query:
        {state["nl_query"]}

        Schema context:
        {state["semantic_context"]}

        Known relationships:
        {state["rels"]}

        Validation issues:
        {state.get('validation_hints','')}
    """

    sql_query = llm_complete(system_prompt,user_prompt.strip()).strip()
    print(f"\n[LLM Raw SQL Attempt #{state['retries'] + 1}] \n{sql_query}")
    state["sql_query"] = sql_query

    # Save the attempt (will be marked as successful later if execution succeeds)
    save_turn(state["nl_query"], sql_query, success=False)  # Mark as tentative
    
    return state

def validator_node(state: PipelineState):
    """
    Node 4: Validates the generated SQL for syntax or basic errors.
    """
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
    """
    Node 5: Executes the SQL query against the database.
    """
    query = state["sql_query"]
    try:
        # Execute the SQL query
        df = run_sql_query(query)
        state["df"] = df
        
        # ✅ Query executed successfully
        print(f"\n✅ SQL Execution Successful!")
        print(f"Result shape: {df.shape if df is not None else 'No data'}")
        
        # Check if this was a correction (error_retries > 0)
        if state.get("error_retries", 0) > 0:
            # This was a successful error correction!
            print("🎯 Successful error correction - storing pattern for learning")
            
            # Try to get the previous broken query from conversation memory
            # This is a simplified approach - you might want to store broken queries explicitly
            broken_query = state.get("previous_broken_query", "Unknown")
            previous_error = state.get("previous_error", "Unknown error")
            
            # Store the successful correction pattern
            add_error_correction_example(
                state["nl_query"], 
                broken_query, 
                previous_error, 
                state["sql_query"]
            )
        
        # Add to few-shot learning (all successful queries)
        add_fewshot_example(state["nl_query"], state["sql_query"])
        
        # Update conversation memory with success
        save_turn(state["nl_query"], state["sql_query"], success=True)
        
        # Clear validation hints
        state["validation_hints"] = None
        
    except Exception as e:
        # ❌ Query execution failed
        error_msg = str(e)
        print(f"\n❌ SQL Execution Failed: {error_msg}")
        
        # Store broken query for potential learning if fixed later
        state["previous_broken_query"] = state["sql_query"]
        state["previous_error"] = error_msg
        
        state["df"] = None
        state["validation_hints"] = error_msg
        
        # Update conversation memory with failure
        save_turn(state["nl_query"], f"FAILED: {error_msg}", success=False)

    return state

def error_handler_node(state: PipelineState):
    """
    Node 6: Advanced error correction using few-shot examples, conversation context, and error patterns.
    """
    # Increment error retry counter
    state["error_retries"] = state.get("error_retries",0) + 1

    system_prompt = (
        f"You are an expert SQL error correction specialist for SQLite.\n"
        f"Use ONLY these exact table names (do NOT singularize or invent names): {state['canonical_tables']}\n"
        "Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE.\n"
        "Take care of values in the table irrespective of user input case (capital or lowercase).\n"
        "Return ONLY the final corrected SELECT SQL (no explanation, no backticks, no markdown formatting).\n"
        "Use provided schema context and join hints. Ensure the SQL is valid SQLite.\n"
        "You may include subqueries or CTEs if needed.\n"
        "Do not use unsupported math functions - use basic aggregate functions instead.\n"
        "Do not use nested aggregate functions.\n"
        "Focus on fixing the specific error while maintaining the original intent.\n"
        "Learn from similar successful examples and error correction patterns.\n"
        "Common SQLite issues to watch for:\n"
        "- Column names with spaces need double quotes\n"
        "- Table names with spaces need backticks or double quotes\n"
        "- JOIN syntax must be explicit\n"
        "- Aggregate functions cannot be nested\n"
        "- LIMIT clause comes at the end\n"
    )

    error_msg = state.get("validation_hints","") or state.get("execution_error","")
    broken_query = state.get("sql_query","")
    nl_query = state.get("nl_query","")
    semantic_context = state.get("semantic_context", "")

    # ---- Get conversation context -------
    conv_context = get_conversation_context(max_turns=2)  # Shorter for error correction
    
    # ------- Get few-shot examples for similar queries ------
    fewshot_prompt = build_fewshot_prompt(nl_query)
    
    # ------- Get specific error correction patterns ------
    error_correction_prompt = build_error_correction_prompt(nl_query, error_msg)
    
    user_prompt = f"""
        Conversation Context:
        {conv_context}

        SUccessful Query Examples:
        {fewshot_prompt}

        Error Correction Patters:
        {error_correction_prompt}

        Here is the available context and schema
        {semantic_context}

        Table Relationships:
        {state.get("rels", [])}

        Failed Query Details:
        User Request: {nl_query}
        Generated SQL: {broken_query}
        Error Message: {error_msg}

        Please carefully review the schema, the original request, the failed query, and the error.
        Then, rewrite the SQL query so it runs correctly on SQLite.
    """
    try:
        # Generate corrected SQL
        fixed_query = llm_complete(system_prompt, user_prompt.strip()).strip()
        
        print(f"\n🔧 ADVANCED ERROR CORRECTION - Attempt #{state['error_retries']}")
        print(f"Query: {nl_query}")
        print(f"Error: {error_msg}")
        print(f"Broken: {broken_query}")
        print(f"Fixed:  {fixed_query}")
        
        state["sql_query"] = fixed_query
        state["validation_hints"] = None  # Clear previous error
        
        # Store this correction attempt for learning
        correction_log = f"CORRECTION #{state['error_retries']} for '{nl_query}': {error_msg}"
        save_turn(correction_log, f"Fixed: {fixed_query}", success=False)
        
    except Exception as e:
        print(f"❌ Error in error_handler_node: {e}")
        state["validation_hints"] = f"Error correction failed: {str(e)}. Original error: {error_msg}"
    
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

# =================================================================
# 4. DEFINE THE GRAPH AND ITS EDGES
# This function assembles the nodes into a coherent workflow.
# =================================================================
def create_graph_app():
    """
    Creates and compiles the LangGraph application.
    This is the function that will be cached by Streamlit.
    """
    MAX_VALIDATION_RETRIES = 2
    MAX_ERROR_RETRIES = 2

    graph = StateGraph(PipelineState)

    # Add all the nodes to the graph
    graph.add_node("retriever", retriever_node)
    graph.add_node("kg", kg_node)
    graph.add_node("llm_sql", llm_sql_node)
    graph.add_node("validator", validator_node)
    graph.add_node("executor", executor_node)
    graph.add_node("error_handler", error_handler_node)
    graph.add_node("visualizer", visualizer_node)

    # Define the starting point of the graph
    graph.add_edge(START, "retriever")

    # Define the linear flow of the graph
    graph.add_edge("retriever", "kg")
    graph.add_edge("kg", "llm_sql")
    graph.add_edge("llm_sql", "validator")

    # --- Define Conditional Edges (Routers) ---
    def validation_router(state: PipelineState):
        if state["is_valid"]:
            return "executor"
        else:
            if state["retries"] + 1 < MAX_VALIDATION_RETRIES:
                state["retries"] += 1
                return "llm_sql" # Go back to regenerate SQL
            else:
                raise ValueError(
                f"Unable to produce valid SQL after {MAX_VALIDATION_RETRIES} attempts. "
                f"Hints: {state['validation_hints']}"
            )
    
    def executor_router(state: PipelineState):
        if state.get("validation_hints"): # An error occurred during execution
            if state.get("error_retries", 0) >= MAX_ERROR_RETRIES:
                raise ValueError("Failed to correct SQL query after multiple attempts.")
            return "error_handler"
        elif state.get("visualize"): # Execution was successful, and user wants a plot
            return "visualizer"
        else: # Execution successful, no visualization needed
            return END
    
    graph.add_conditional_edges(
        "validator",
        validation_router,
        {"executor": "executor", "llm_sql": "llm_sql"}
    )
      
    graph.add_conditional_edges(
        "executor",
        executor_router,
        {"error_handler": "error_handler", "visualizer": "visualizer", END: END}
    )

    # Define final edges
    graph.add_edge("error_handler", "executor") # After fixing, retry execution
    graph.add_edge("visualizer", END)           # After visualizing, end

    # Compile the graph into a runnable application
    app = graph.compile()
    return app