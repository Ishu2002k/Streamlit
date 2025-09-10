import sqlite3
import re
import os
import sqlitecloud 
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# ------------------ Utility ------------------
def quote_identifier(name: str) -> str:
    """Safely quote SQL identifiers (tables/columns)."""
    return f'"{name}"'

def guess_column_type(values):
    """
    Infer a more semantic column type from sample values.
    Args:
        values (list): list of sample values from a column
    Returns:
        str: inferred type (DATE, DATETIME, INTEGER, FLOAT, TEXT, etc.)
    """
    values = [v for v in values if v is not None]
    if not values:
        return "TEXT"
    
    # Check Boolean
    if all(str(v).lower() in ["true", "false", "0", "1"] for v in values[:30]):
        return "BOOLEAN"

    # Check integer
    if all(re.match(r"^-?\d+$", str(v)) for v in values[:30]):
        return "INTEGER"

    # Check float
    if all(re.match(r"^-?\d+(\.\d+)?$", str(v)) for v in values[:30]):
        return "FLOAT"

    # Check date formats
    date_patterns = [
        r"^\d{4}-\d{2}-\d{2}$",  # 2023-09-04
        r"^\d{2}/\d{2}/\d{4}$",  # 09/04/2023
    ]
    if all(any(re.match(p, str(v)) for p in date_patterns) for v in values[:30]):
        return "DATE"

    # Check datetime
    try:
        for v in values[:30]:
            datetime.fromisoformat(str(v))
        return "DATETIME"
    except Exception:
        pass

    return "TEXT"

def fetch_table_name(db_path):
    conn = sqlitecloud.connect(db_path)
    cursor = conn.cursor()

    # 1. Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' and name NOT LIKE '_sqlite%';")
    tables = [row[0] for row in cursor.fetchall()]

    return tables

def extract_schema(db_path,sample_limit = 200,example_limit = 3):
    """
    Extracts schema info, column details, and relationships from SQLite database.
    Args:
        db_path (str): Path to SQLite database file or connection string.
    Returns:
        dict: schema with tables, columns, relationships
    """
    conn = sqlitecloud.connect(db_path)
    cursor = conn.cursor()
    schema_info = {"tables": {}, "relationships": []}

    # 1. Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' and name NOT LIKE '_sqlite%';")
    tables = [row[0] for row in cursor.fetchall()]

    for table in tables:
        schema_info["tables"][table] = {"columns": [],"examples":[]}

        # 2. Get column details
        cursor.execute(f"PRAGMA table_info({quote_identifier(table)});")
        # Five values: (Index, Column Name, Data Type, Not Null, Default Value, Primary Key)
        columns = cursor.fetchall()

        for col in columns:
            col_name, declared_type,pk_flag = col[1], col[2],col[5]

            # --- NEW: sample values to refine datatype ---
            cursor.execute(f"SELECT {quote_identifier(col_name)} FROM {quote_identifier(table)} LIMIT 30;")
            sample_values = [row[0] for row in cursor.fetchall()]
            inferred_type = guess_column_type(sample_values)
            # prefer inferred type if it is more meaningful
            col_type = inferred_type if inferred_type != "TEXT" else declared_type

            # Check for uniqueness if not already a primary key
            is_candidate_pk = False
            if pk_flag == 0:
                try:
                    cursor.execute(f"""
                        SELECT COUNT(*) = COUNT(DISTINCT {quote_identifier(col_name)})
                        FROM {quote_identifier(table)}
                        WHERE {quote_identifier(col_name)} IS NOT NULL;
                    """)
                    is_candidate_pk = (cursor.fetchone()[0] == 1)
                except Exception:
                    pass

            schema_info["tables"][table]["columns"].append(
                {"name": col_name, 
                 "type": col_type,
                 "is_primary_key": pk_flag == 1,
                 "is_candidate_pk": is_candidate_pk
            })

        try:
            cursor.execute(f"select * from {quote_identifier(table)} LIMIT {example_limit};")
            rows = cursor.fetchall()
            col_names = [desc[0] for desc in cursor.description]
            for row in rows:
                schema_info["tables"][table]["examples"].append(
                    dict(zip(col_names,row))
                )
        except Exception:
            pass

        # 3. Get foreign key constraints (if explicitly defined)
        cursor.execute(f"PRAGMA foreign_key_list({quote_identifier(table)});")
        fks = cursor.fetchall()
        for fk in fks:
            schema_info["relationships"].append({
                "from_table": table,
                "from_column": fk[3],
                "to_table": fk[2],
                "to_column": fk[4],
                "type": "foreign_key"
            })

    for from_table in tables:
        from_columns = schema_info['tables'][from_table]["columns"]

        for from_col in from_columns:
            col_name = from_col["name"]
            col_type = from_col["type"]

            # Heuristics: column ends with id
            if(col_name.endswith("_id")):
                base_name = col_name.replace("_id","")
                candidates = [base_name, base_name + "s"]

                for to_table in tables:
                    if to_table in candidates:
                        to_columns = schema_info["tables"][to_table]["columns"]
                        to_col_names = [c["name"] for c in to_columns]
                        if "id" in to_col_names:
                            schema_info["relationships"].append({
                                "from_table": from_table,
                                "from_column": col_name,
                                "to_table": to_table,
                                "to_column": "id",
                                "type": "name_heuristic"
                            })
            
            # Type-based + value overlap check
            for to_table in tables:
                if to_table == from_table:
                    continue

                to_columns = schema_info["tables"][to_table]["columns"]
                for to_col in to_columns:
                    if to_col["type"] == col_type:
                        # Sample values from both columns
                        try:
                            cursor.execute(f"""
                                SELECT DISTINCT {col_name} FROM \"{from_table}\"
                                WHERE {col_name} IS NOT NULL LIMIT {sample_limit}
                            """)
                            from_values = set(row[0] for row in cursor.fetchall())

                            cursor.execute(f"""
                                SELECT DISTINCT {to_col['name']} FROM \"{to_table}\"
                                WHERE {to_col['name']} IS NOT NULL LIMIT {sample_limit}
                            """)
                            to_values = set(row[0] for row in cursor.fetchall())

                            overlap = from_values & to_values
                            if len(overlap) > 0 and len(overlap) / max(1, len(from_values)) > 0.5:
                                schema_info["relationships"].append({
                                    "from_table": from_table,
                                    "from_column": col_name,
                                    "to_table": to_table,
                                    "to_column": to_col["name"],
                                    "type": "value_overlap"
                                })
                        except Exception:
                            continue
                        
    conn.close()
    return schema_info

def generate_embedding_text(schema_info,sample_rows = 5):
    """
    Generate descriptive text from schema for embeddings.
    Args:
        schema_info (dict): schema extracted from extract_schema
    Returns:
        list of dict: items with id, text
    """
    embedding_docs = []

    # Table-level text
    for table, details in schema_info["tables"].items():
        cols = ", ".join([f"{c['name']} ({c['type']})" for c in details["columns"]])
        text = f"Table: {table}. Columns: {cols}."
        embedding_docs.append({"id": f"table::{table}", "text": text})

        # Column-level text
        for col in details["columns"]:
            text = f"Column: {col['name']} in {table}. Type: {col['type']}."
            embedding_docs.append({"id": f"column::{table}.{col['name']}", "text": text})

    # Relationship-level text
    for rel in schema_info["relationships"]:
        text = f"Relationship: {rel['from_table']}.{rel['from_column']} -> {rel['to_table']}.{rel['to_column']} ({rel['type']})."
        embedding_docs.append({"id": f"rel::{rel['from_table']}.{rel['from_column']}->{rel['to_table']}.{rel['to_column']}", "text": text})

    return embedding_docs

# Example usage
if __name__ == "__main__":
    schema = extract_schema(db_path)
    embedding_docs = generate_embedding_text(schema)

    print("Schema Info:", schema)
    print("\nEmbedding Texts:")
    for doc in embedding_docs:
        print(doc)