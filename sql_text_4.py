import re
from typing import Tuple, Dict, List
from step_1 import extract_schema, generate_embedding_text
from sql_txt_2 import find_join_path,build_schema_kg,build_faiss_vectorstore,generate_embeddings_hf
from sql_text_3 import llm_complete,run_sql_query
from dotenv import load_dotenv
import os
from sqlglot import exp,parse_one
import networkx as nx
import json
from sqlglot.expressions import Column, Alias

# -------- Load env --------
load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# -----------------------
# KG helpers / normalizers
# -----------------------
def get_table_names_from_kg(kg: nx.DiGraph) -> dict[str, str]:
    """
    Returns a dictionary mapping lowercase table names -> canonical table names.
    Only considers nodes that are tuples starting with "table".
    Safely handles missing or empty data attributes.
    """
    return {
        node[1].lower(): node[1]
        for node, data in kg.nodes(data=True)
        if isinstance(node, tuple) and node[0] == "table"
    }

def get_columns_for_table_from_kg(kg: nx.DiGraph) -> Dict[str,set]:
    """
    Returns: dict canonical_table -> set(lowercase_column_names)
    """
    cols = {}
    for node, data in kg.nodes(data=True):
        if isinstance(node, tuple) and node[0] == "column":
            _, table, column = node
            cols.setdefault(table, set()).add(column.lower())
    # ensure tables present even if no columns found
    for t in list(get_table_names_from_kg(kg).values()):
        cols.setdefault(t, set())
    return cols

def canonical_table_name(kg, name_candidate: str):
    """Map a name (alias or table name) to canonical table name using KG (case-insensitive)."""
    name_l = name_candidate.strip().strip('"').strip("'").lower()
    tables_map = get_table_names_from_kg(kg)
    return tables_map.get(name_l)  # None if not found

# -----------------------
# Validator using KG
# -----------------------
def validate_sql_with_kg(sql_query: str, kg, max_suggest: int = 3) -> Tuple[bool, List[str]]:
    """
    Validate a SQL query against the Knowledge Graph (KG), including alias handling.
    """
    hints = []
    valid = True

    try:
        parsed = parse_one(sql_query, read="sqlite")
    except Exception as e:
        return False, [f"❌ SQL parsing failed: {e}"]

    # --- Extract CTE names ---
    cte_names = {
        cte.alias_or_name.lower()
        for with_expr in parsed.find_all(exp.With)
        for cte in with_expr.expressions
        if cte.alias_or_name
    }

    # --- Map aliases to tables or CTEs ---
    alias_to_table = {}
    for table_expr in parsed.find_all(exp.Table):
        alias = table_expr.alias_or_name
        table_name = table_expr.name
        if alias:
            alias_to_table[alias.lower()] = table_name.lower()
        else:
            alias_to_table[table_name.lower()] = table_name.lower()

    for from_or_join in list(parsed.find_all(exp.From)) + list(parsed.find_all(exp.Join)):
        for source in from_or_join.expressions:
            alias = source.alias_or_name
            if alias:
                if isinstance(source.this, exp.Identifier):
                    target_name = source.this.name
                elif isinstance(source.this, exp.Table):
                    target_name = source.this.name
                else:
                    target_name = None
                if target_name:
                    alias_to_table[alias.lower()] = target_name.lower()

    # --- Prepare KG maps ---
    table_map = get_table_names_from_kg(kg)
    cols_by_table = get_columns_for_table_from_kg(kg)
    canonical_tables = set(table_map.values())

    # --- Collect derived column aliases ---
    derived_aliases = set()
    for subq in [parsed] + list(parsed.find_all(exp.Subquery)):
        for alias_expr in subq.find_all(Alias):
            if alias_expr.alias:
                derived_aliases.add(alias_expr.alias.lower())

    # --- Extract columns ---
    qualified_columns = []
    unqualified_columns = []
    for col in parsed.find_all(Column):
        if col.name == "*":
            continue
        if col.table:
            qualified_columns.append((col.table.lower(), col.name))
        else:
            unqualified_columns.append(col.name)

    # --- Validate qualified columns ---
    for alias, col in qualified_columns:
        col_l = col.lower()
        if col_l in derived_aliases:
            continue  # Skip derived columns

        if alias in alias_to_table:
            resolved_table = canonical_table_name(kg, alias_to_table[alias]) or alias_to_table[alias]
        elif alias in cte_names:
            resolved_table = alias
        else:
            resolved_table = canonical_table_name(kg, alias) or alias

        if resolved_table in cols_by_table:
            if col_l not in cols_by_table[resolved_table]:
                candidates = [t for t, cols in cols_by_table.items() if col_l in cols]
                if candidates:
                    suggestions = []
                    for cand in candidates[:max_suggest]:
                        path = find_join_path(kg, resolved_table, cand)
                        suggestions.append(f"{cand} (join: {path['path_str'] if path else 'no direct path'})")
                    hints.append(
                        f"Column '{col}' not found in table '{resolved_table}'. "
                        f"Found in: {', '.join(candidates[:max_suggest])}. "
                        f"Suggestions: {', '.join(suggestions)}"
                    )
                else:
                    hints.append(
                        f"Column '{col}' not found in table '{resolved_table}', "
                        f"and no other table contains it."
                    )
                valid = False
        elif resolved_table in cte_names:
            continue
        else:
            hints.append(
                f"Referenced table/alias '{alias}' could not be resolved. "
                f"Known tables: {sorted(canonical_tables)} + CTEs: {sorted(cte_names)}"
            )
            valid = False

    # --- Validate unqualified columns ---
    source_tables = set(alias_to_table.values())
    source_canonicals = [canonical_table_name(kg, t) or t for t in source_tables] or list(cols_by_table.keys())

    for col in unqualified_columns:
        col_l = col.lower()
        if col_l in derived_aliases:
            continue  # Skip derived columns

        matches = [t for t in source_canonicals if col_l in cols_by_table.get(t, set())]

        if len(matches) == 1:
            continue
        elif len(matches) > 1:
            hints.append(f"Unqualified column '{col}' is ambiguous among tables {matches}. Please qualify it.")
            valid = False
        else:
            found_in = [t for t, cols in cols_by_table.items() if col_l in cols]
            if found_in:
                suggestions = []
                for cand in found_in[:max_suggest]:
                    start = source_canonicals[0] if source_canonicals else cand
                    path = find_join_path(kg, start, cand)
                    suggestions.append(f"{cand} (path: {path['path_str'] if path else 'no path'})")
                hints.append(
                    f"Column '{col}' not found in source tables {source_canonicals}. "
                    f"Found in: {found_in[:max_suggest]}. "
                    f"Suggested joins: {suggestions}"
                )
            else:
                hints.append(f"Column '{col}' not found in any table.")
            valid = False

    return valid, hints if hints else ["✅ SQL query is valid"]

# -----------------------
# Integrate into answer_query
# -----------------------
def answer_query_with_validation(nl_query, vectorstore, kg, max_retries=2):
    """
    Full pipeline with sqlglot:
      1. RAG retrieval -> semantic_context
      2. LLM suggests structure (tables, columns, filters, joins)
      3. Build actual SQL using sqlglot (safe AST-based)
      4. validate_sql_with_kg -> hints if invalid
      5. Retry with structured hints if needed
      6. Run final validated SQL
    """
    # 1. Retrieve semantic context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    retrieved_docs = retriever.invoke(nl_query)
    semantic_context = "\n".join([doc.page_content for doc in retrieved_docs])

    # 2. Prepare KG metadata
    table_map = get_table_names_from_kg(kg)
    canonical_tables = list(table_map.values())

    rels = []
    for u, v, data in kg.edges(data=True):
        if isinstance(u, tuple) and isinstance(v, tuple) and u[0] == "table" and v[0] == "table":
            rels.append(f"{u[1]} -> {v[1]} ({data.get('relation')})")

    # 3. LLM prompt for structured plan
    system_prompt = (
        f"You are an expert SQL generator for SQLite.\n"
        "Do not use any alias in SQL Query\n"
        f"Use ONLY these exact table names (do NOT singularize or invent names): {canonical_tables}\n"
        "Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE.\n"
        "Take care of values in the table irrespective user asked in capital or in lower case.\n"
        "Return ONLY the final SELECT SQL (no explanation, no backticks).\n"
        "Use provided schema context and join hints. Ensure the SQL is valid SQLite.\n"
        "You may include subqueries or CTEs using nested JSON objects.\n"
        "Use right math function if needed.\n"
        "Do not use nested aggregate functions\n"
    )

    user_prompt = f"""
    NL query:
    {nl_query}

    Schema context:
    {semantic_context}

    Known relationships:
    {rels}
    """

    retries = 0
    while retries < max_retries:
        sql_query = llm_complete(system_prompt, user_prompt).strip()
        print(f"\n[LLM Raw SQL Attempt {retries+1}]\n{sql_query}")

        try:
            parsed = parse_one(sql_query, dialect="sqlite")
        except Exception as e:
            print(f"\n❌ sqlglot parsing failed: {e}")
            retries += 1
            continue

        is_valid, hints = validate_sql_with_kg(sql_query, kg)
        if is_valid:
            break

        print(f"\n[KG Validation Failed]\n{hints}")
        user_prompt += f"\n\nValidation issues:\n{hints}\nPlease regenerate the SQL."

        retries += 1

    if not is_valid:
        raise ValueError(f"Unable to produce valid SQL after {max_retries} attempts. Hints: {hints}")

    # 6. Execute SQL
    df = run_sql_query(sql_query)
    print(f"\n✅ Final SQL Query:\n{sql_query}")
    print(f"\n📊 Output:\n{df}")

    return sql_query, df

# ------------------- MAIN --------------------
if __name__ == "__main__":
    # Extract schema
    schema_info = extract_schema(db_path)

    # Embeddings
    embedding_docs = generate_embedding_text(schema_info)
    embedded_docs = generate_embeddings_hf(embedding_docs)

    # Vectorstore (FAISS or Chroma)
    vectorstore = build_faiss_vectorstore(embedded_docs)

    # Build Knowledge Graph
    G = build_schema_kg(schema_info)

    # Example NL query
    # nl_query = "Find all the People with region they belong who have returned atleast one order"
    # nl_query = "Detect outlier orders: orders where the amount is greater than mean + 2*stddev for that Region."
    # nl_query = "List People who never returned an order but belong to Regions where return rate > 30%."
    # nl_query = "Find the month with the highest return rate across all orders"
    # nl_query = "For each Region, calculate the percentage contribution of each Person’s spending to the Region’s total."
    nl_query = "Find the Person_Name who has spent the maximum total Amount"
    answer_query_with_validation(nl_query, vectorstore, G)