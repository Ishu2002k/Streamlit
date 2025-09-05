import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores import FAISS, Chroma
import networkx as nx
from step_1 import extract_schema, generate_embedding_text
from step_2 import build_knowledge_graph, generate_embeddings_hf, build_chroma_vectorstore

# Load env
load_dotenv()

# -----------------------------
# Step 2: Initialize Embedding Model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")
)

# -----------------------------
# Vector Search + KG traversal
# -----------------------------
def search_schema(query, vectorstore, kg_graph, top_k=5):
    """
    Returns minimal relevant schema context for a user query.
    
    Args:
        query (str): user natural language query
        vectorstore: FAISS or Chroma vector store
        kg_graph: NetworkX graph of schema
        top_k: number of top semantic hits from vector search
    Returns:
        context_text (str): text description of relevant tables/columns + relationships
        involved_tables (list): list of relevant tables after KG traversal
    """
    # --- Step 1: Semantic search in vector store ---
    results = vectorstore.similarity_search(query, k=top_k)
    
    # --- Step 2: Identify tables and columns from retrieved docs ---
    retrieved_nodes = set()
    for r in results:
        meta_id = r.metadata.get("id", "")
        if meta_id.startswith("table::"):
            retrieved_nodes.add(meta_id.replace("table::", ""))
        elif meta_id.startswith("column::"):
            retrieved_nodes.add(meta_id.replace("column::", ""))

    # --- Step 3: KG Traversal to find connected tables ---
    tables_involved = set()
    for node in retrieved_nodes:
        if "." in node:  # column node
            table_name = node.split(".")[0]
            tables_involved.add(table_name)
        elif node in kg_graph.nodes and kg_graph.nodes[node].get("type") == "table":
            tables_involved.add(node)

    # Expand via KG edges (join paths)
    expanded_tables = set(tables_involved)
    for table in tables_involved:
        for neighbor in nx.single_source_shortest_path_length(kg_graph, table, cutoff=2):
            if kg_graph.nodes[neighbor].get("type") == "table":
                expanded_tables.add(neighbor)

    # --- Step 4: Compose schema context text ---
    context_lines = []
    for node in expanded_tables:
        context_lines.append(f"Table: {node}")
        for col in kg_graph.successors(node):
            if kg_graph.nodes[col]["type"] == "column":
                dtype = kg_graph.nodes[col]["data_type"]
                context_lines.append(f"Column: {col.split('.')[-1]} ({dtype})")
    
    # Include relationships among selected columns
    for u, v, d in kg_graph.edges(data=True):
        if u in retrieved_nodes or v in retrieved_nodes:
            relation = d.get("relation")
            context_lines.append(f"Relationship: {u} -> {v} ({relation})")

    context_text = "\n".join(context_lines)
    return context_text, list(expanded_tables)


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # --- Step 1: Extract schema ---
    db_path = os.getenv("DATABASE_CONNECTION_STRING")
    schema_info = extract_schema(db_path)

    # --- Step 2: Build Knowledge Graph ---
    kg_graph = build_knowledge_graph(schema_info)

    # --- Step 3: Generate embeddings and store in Chroma ---
    embedding_docs = generate_embedding_text(schema_info)
    docs = generate_embeddings_hf(embedding_docs)
    chroma_store = build_chroma_vectorstore(docs)

    # --- Step 4: User query ---
    user_query = "Find all the person name from central region"
    context_text, involved_tables = search_schema(user_query, chroma_store, kg_graph, top_k=5)

    print("\n--- Relevant Schema Context ---")
    print(context_text)
    print("\n--- Tables Involved ---")
    print(involved_tables)
