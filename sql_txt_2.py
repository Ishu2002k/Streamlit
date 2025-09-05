import os
import numpy as np
import plotly.express as px
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from sqlglot import parse_one
from langchain_huggingface import HuggingFaceEmbeddings
from openai import AzureOpenAI
import networkx as nx
from networkx.exception import NetworkXNoPath, NodeNotFound
from langchain_community.vectorstores import FAISS, Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document

# --------------- Load env ---------------
load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# Initialize HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")
)

def build_faiss_vectorstore(docs):
    """Build FAISS in-memory vectorspace"""
    return FAISS.from_documents(docs,embedding_model)

def build_chroma_vectorstore(docs,persist_dir = "./chroma_db"):
    """Build Chroma persistent vectorstore"""
    vectorstore = Chroma.from_documents(
                            docs,
                            embedding_model,
                            persist_directory = persist_dir
                            )

    return vectorstore

# ---------- Step 1 imports you already have ----------
# from step_1 import extract_schema, generate_embedding_text
from step_1 import extract_schema, generate_embedding_text

# --------------- Embedding Model (BGE) ----------------
EMB_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")

# ---------------- LLM Caller (Azure Compatible) ------------
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# ---------- Knowledge Graph (tables/columns/edges) ----------
def quote_identifier(identifier):
    return f'"{identifier}"'

def build_schema_kg(schema_info):
    """
    Build a knowledge graph using NetworkX from schema_info.
    
    Args:
        schema_info (dict): Extracted schema with tables, columns, relationships.
        
    Returns:
        nx.Graph: Knowledge graph of schema
    """
    G = nx.DiGraph()

    # --------- Add table and Column Nodes ---------
    for table, details in schema_info["tables"].items():
        table_node = quote_identifier(table)
        G.add_node(table,type = "table",label = table)

        for col in details["columns"]:
            col_node = f"{table_node}.{quote_identifier(col['name'])}"
            G.add_node(col_node,type = "column",dtype = col["type"],label = col["name"])
            G.add_edge(table_node,col_node,relation = "has_column")

    # ---------- Add relationships -------------
    for rel in schema_info.get("relationships",[]):
        if(all(k in rel for k in ("from_table","from_column","to_table","to_column","type"))):
            from_col = f"{quote_identifier(rel['from_table'])}.{quote_identifier(rel['from_column'])}"
            to_col = f"{quote_identifier(rel['to_table'])}.{quote_identifier(rel['to_column'])}"
            G.add_edge(from_col,to_col,relation = rel["type"],label = rel["type"])
        
        else:
            print(f"⚠️ Skipping incomplete relationship: {rel}")
    
    # ----------- Add row-level examples ---------------
    for example in schema_info.get("examples", []):
        table = quote_identifier(example["table"])
        row_id = quote_identifier(example.get("id", f"{table}_row_{hash(str(example)) % 10000}"))
        row_node = f"{table}.{row_id}"
        G.add_node(row_node, type="row", label=row_id)

        for col_name, value in example["values"].items():
            col_node = f"{table}.{quote_identifier(col_name)}"
            value_node = f"{row_node}.{quote_identifier(col_name)}"
            G.add_node(value_node, type="value", label=str(value))
            G.add_edge(row_node, value_node, relation="has_value", column=col_name)
            G.add_edge(value_node, col_node, relation="instance_of", label="instance_of")

    return G

def find_join_path(G,source_table,target_table):
    """
    Find the shortest join path between two tables.
    
    Args:
        G (nx.Graph): Knowledge graph
        source_table (str): Starting table
        target_table (str): Target table
    
    Returns:
        list: Path of tables/columns or None if no path exists
    """
    source_node = quote_identifier(source_table)
    target_node = quote_identifier(target_table)
    try:
        path = nx.shortest_path(G,source = source_node,target = target_node)
        return " -> ".join(path)
    except (NetworkXNoPath, NodeNotFound):
        return None


def generate_embeddings_hf(embedding_docs):
    """
    Generate embeddings for schema descriptions as Document objects.
    Only store 'id' in metadata, NOT the embedding itself.
    """
    docs = []
    for doc in embedding_docs:
        try:
            docs.append(Document(
                page_content=doc["text"],
                metadata={"id": doc["id"]}  # embeddings handled separately
            ))
        except Exception as e:
            print(f"Error creating document for ID {doc['id']}: {e}")
    return docs

# ---------- SQL Generation Prompt ----------
SQL_SYSTEM = (
    "You are a Text-to-SQL assistant. "
    "Generate a **SELECT-only** SQL query for SQLite. "
    "Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE."
    "Generate valid SQL query for the user query."
    "Use provided schema context and relationships."
    "Take care of values in the table irrespective user asked in capital or in lower case"
    "Do not explain. No backticks."
)

SQL_USER_TMPL = """User question:
    {question}

    Relevant schema snippets:
    {schema_snips}

    Known relationships:
    {rels}

    Rules:
    - Prefer explicit JOINs.
    - Use valid SQLite syntax.
    # - Limit rows to 200 unless user requests more.
    - Return only the SQL query.
    """

import matplotlib.pyplot as plt
import networkx as nx

def visualize_kg(G):
    pos = nx.spring_layout(G, seed=42)  # Layout algorithm
    node_labels = nx.get_node_attributes(G, 'label')
    edge_labels = nx.get_edge_attributes(G, 'relation')

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=800, node_color='skyblue', font_size=10, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray')
    plt.title("Knowledge Graph Visualization")
    plt.show()


if __name__ == "__main__":
    schema_info = extract_schema(db_path)
    embedding_docs = generate_embedding_text(schema_info)
    embedded_docs = generate_embeddings_hf(embedding_docs)
    G = build_schema_kg(schema_info)
    visualize_kg(G)
