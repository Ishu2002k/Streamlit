import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from openai import AzureOpenAI
import networkx as nx
from networkx.exception import NetworkXNoPath, NodeNotFound
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
import matplotlib.pyplot as plt
from step_1 import extract_schema, generate_embedding_text

# --------------- Load env ---------------
load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# --------------- Embedding Model (BGE) ----------------
EMB_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
LLM_MODEL = os.getenv("LLM_MODEL")

# ---------------- LLM Caller (Azure Compatible) ------------
client = AzureOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("OPENAI_API_VERSION"),
)

# ---------- Initialize HuggingFace Embeddings -------------
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")
)

# ------------- Generate Embedding from docs ----------------
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

# ---------- Knowledge Graph (tables/columns/edges) ----------
def quote_identifier(identifier):
    return f'"{identifier}"'

# --------- Tuple Based Information --------------
def build_schema_kg(schema_info):
    """
    Build a knowledge graph using NetworkX from schema_info.
    Node format:
      ("table", table_name)
      ("column", table_name, column_name)
    Args:
        schema_info (dict): Extracted schema with tables, columns, relationships.
    Returns:
        nx.DiGraph: Knowledge graph of schema
    """
    G = nx.DiGraph()

    # --------- Add table and Column Nodes ---------
    for table, details in schema_info["tables"].items():
        table_node = ("table", table)
        G.add_node(table_node, label=table)

        for col in details["columns"]:
            col_node = ("column", table, col["name"])
            G.add_node(col_node, dtype=col["type"], label=col["name"])
            # link: table -> column
            G.add_edge(table_node, col_node, relation="has_column")

    # ---------- Add relationships -------------
    for rel in schema_info.get("relationships", []):
        if all(k in rel for k in ("from_table", "from_column", "to_table", "to_column", "type")):
            from_table = ("table", rel["from_table"])
            to_table = ("table", rel["to_table"])
            from_col = ("column", rel["from_table"], rel["from_column"])
            to_col = ("column", rel["to_table"], rel["to_column"])

            # column -> column edge
            G.add_edge(from_col, to_col, relation=rel["type"], label=rel["type"])
            G.add_edge(to_col, from_col, relation=rel["type"], label=rel["type"] + "_rev") 

            # table -> table edge
            G.add_edge(from_table, to_table, relation=f"{rel['type']}_table")
            G.add_edge(to_table, from_table, relation=f"{rel['type']}_table_rev")
        else:
            print(f"⚠️ Skipping incomplete relationship: {rel}")

    return G

def find_join_path(G, source_table, target_table):
    """
    Find the shortest join path between two tables in the KG.
    Args:
        G (nx.Graph): Knowledge graph
        source_table (str): Starting table name
        target_table (str): Target table name
    Returns:
        dict | None: {
            "nodes": [list of tuple nodes in path],
            "path_str": readable string representation
        }
        or None if no path exists
    """
    source_node = ("table", source_table)
    target_node = ("table", target_table)

    try:
        nodes = nx.shortest_path(G, source=source_node, target=target_node)

        # Convert tuple nodes into nice readable labels
        def fmt(node):
            if node[0] == "table":
                return f"[TABLE] {node[1]}"
            elif node[0] == "column":
                return f"[COLUMN] {node[1]}.{node[2]}"
            else:
                return str(node)

        return {
            "nodes": nodes,
            "path_str": " -> ".join(fmt(n) for n in nodes)
        }

    except (NetworkXNoPath, NodeNotFound):
        return None

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
