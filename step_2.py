import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
import networkx as nx
from step_1 import extract_schema, generate_embedding_text

# Load environment variables
load_dotenv()

# Initialize HuggingFace Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=os.getenv("MODEL_NAME", "BAAI/bge-base-en-v1.5")
)

# -----------------------------
# Step 1: Build Vector Stores
# -----------------------------

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

# -----------------------------
# Step 2: Build Knowledge Graph
# -----------------------------
def build_knowledge_graph(schema_info):
    """
    Build a NetworkX graph with:
    - Nodes = tables & columns
    - Edges = foreign keys or inferred relationships
    """
    G = nx.DiGraph()

    # Add table nodes
    for table,details in schema_info["tables"].items():
        G.add_node(table, type="table")
        for col in details["columns"]:
            col_id = f"{table}.{col['name']}"
            G.add_node(col_id, type="column", data_type=col["type"])
            G.add_edge(table, col_id, relation="has_column")
    
    # Add relationships
    for rel in schema_info["relationships"]:
        from_col = f"{rel['from_table']}.{rel['from_column']}"
        to_col = f"{rel['to_table']}.{rel['to_column']}"
        G.add_edge(from_col, to_col, relation=rel["type"])

    return G


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


def plot_embeddings_2d(embedded_docs, method="tsne"):
    """
    Reduce embeddings to 2D and plot clusters.
    
    Args:
        embedded_docs (list of dict): [{"id": ..., "text": ..., "embedding": [...]}]
        method (str): "tsne" or "pca"
    """
    embeddings = np.array([doc["embedding"] for doc in embedded_docs])
    labels = [doc["id"] for doc in embedded_docs]

    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=10)
    else:
        reducer = PCA(n_components=2)
    reduced = reducer.fit_transform(embeddings)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced[:, 0], reduced[:, 1], c='blue', alpha=0.6)

    # Annotate points with IDs
    for i, label in enumerate(labels):
        plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8, alpha=0.7)

    plt.title(f"2D {method.upper()} visualization of schema embeddings")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.show()

# -----------------------------
# Full Pipeline
# -----------------------------
if __name__ == "__main__":
    # Extract schema
    db_path = os.getenv("DATABASE_CONNECTION_STRING")
    schema_info = extract_schema(db_path)
    # print(schema_info)

    # Generate embedding text
    embedding_docs = generate_embedding_text(schema_info)
    # print(len(embedding_docs))
    # for docs in embedding_docs:
    #     print(docs)
    #     print("\n---------------------\n")

    # # Generate embeddings
    embedded_docs = generate_embeddings_hf(embedding_docs)
    # print(len(embedded_docs))

    # # Store in FAISS
    faiss_store = build_faiss_vectorstore(embedded_docs)
    print(f"FAISS vectorstore created with {len(embedded_docs)} embeddings.")
    # print(faiss_store)

    # # Store in Chroma
    chroma_store = build_chroma_vectorstore(embedded_docs)
    print(f"Chroma vectorstore persisted at './chroma_db'.")

    # # Build Knowledge Graph
    KG = build_knowledge_graph(schema_info)
    print(f"Knowledge Graph built with {KG.number_of_nodes()} nodes and {KG.number_of_edges()} edges.")

    # # Example: show all tables
    tables = [n for n, attr in KG.nodes(data=True) if attr["type"] == "table"]
    print("Tables in KG:", tables)

    # # Example: show relationships for a specific column
    example_col = f"{tables[0]}.{KG.nodes[tables[0]]['type']}"
    print("Example column relationships:", list(KG.edges(tables[0], data=True)))
          
    # Show some sample results
    # print("\nSample Embeddings:")
    # for doc in embedded_docs[:3]:  # just the first 3 for readability
    #     print(f"ID: {doc['id']}")
    #     print(f"Text: {doc['text']}")
    #     print(f"Embedding (first 10 values): {np.array(doc['embedding'])[:10]}")  # first 10 numbers
    #     print(f"Embedding Dimension: {len(doc['embedding'])}\n")

    # # Plot embeddings in 2D using t-SNE
    # # plot_embeddings_2d(embedded_docs, method="tsne")
    
    # # Optional: also try PCA
    # plot_embeddings_2d(embedded_docs, method="pca")
