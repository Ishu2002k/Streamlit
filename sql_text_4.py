import re
from typing import Tuple, Dict, List
from step_1 import extract_schema, generate_embedding_text,fetch_table_name
from sql_txt_2 import find_join_path,build_schema_kg,build_faiss_vectorstore,build_chroma_vectorstore,generate_embeddings_hf
from sql_text_3 import llm_complete,run_sql_query
from dotenv import load_dotenv
import os
from sqlglot import exp,parse_one
import networkx as nx
import json

# -------- Load env --------
load_dotenv()
db_path = os.getenv("DATABASE_CONNECTION_STRING")

# -----------------------
# KG helpers / normalizers
# -----------------------
# def get_table_names_from_kg(kg: nx.DiGraph) -> Dict[str,str]:
#     """
#     Returns: dict mapping lowercase -> canonical table name
#     """
#     tables = {}
#     for node, data in kg.nodes(data=True):
#         # adapted for tuple-based nodes: ("table", table_name)
#         if isinstance(node, tuple) and node[0] == "table":
#             name = node[1]
#             tables[name.lower()] = name
#     return tables  # lower -> canonical

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
# SQL parsing (basic)
# -----------------------
def parse_sql_references(sql: str) -> Dict:
    """
    Returns:
      {
        "alias_to_table": { alias_lower: canonical_table_name_candidate_or_table_literal },
        "qualified_columns": [(table_or_alias, column)],
        "select_unqualified": [col1, col2, ...]   # columns in SELECT without qualifier
      }
    NOTE: this is a pragmatic regex-based extractor (handles common SELECT ... FROM ... JOIN alias patterns)
    """
    sql_text = sql.strip()
    # 1) find FROM/JOIN table and alias pairs
    alias_to_table = {}
    for match in re.finditer(r"(?:FROM|JOIN)\s+\"?([A-Za-z0-9_]+)\"?\s*(?:AS\s+)?(?:(\b[A-Za-z0-9_]+\b))?", sql_text, flags=re.I):
        table_name, alias = match.group(1), match.group(2)
        key = (alias or table_name).lower()
        alias_to_table[key] = table_name  # value is the literal table name as in SQL

    # 2) find qualified columns like alias.col or Table.col
    qualified = re.findall(r"([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)", sql_text)
    # This will find many things; keep as-is and resolve later
    qualified_columns = [(t, c) for (t, c) in qualified]

    # 3) extract SELECT … FROM part to capture unqualified select columns
    select_unqualified = []
    m = re.search(r"SELECT\s+(.*?)\s+FROM\s", sql_text, flags=re.I | re.S)
    if m:
        select_list = m.group(1)
        # split by comma respecting parentheses simply enough
        parts = [p.strip() for p in re.split(r",(?![^\(\)]*\))", select_list)]
        for p in parts:
            # remove aliases in select (AS alias)
            p_clean = re.sub(r"\s+AS\s+.+$", "", p, flags=re.I).strip()
            # if it's qualified skip (we handle qualified separately)
            if "." in p_clean:
                continue
            # remove function calls, keep bare identifier if present at end
            # e.g., COUNT(order_id) -> no direct mapping here; skip
            mcol = re.search(r"([A-Za-z0-9_]+)$", p_clean)
            if mcol:
                colname = mcol.group(1)
                select_unqualified.append(colname)
    # Return info
    return {
        "alias_to_table": alias_to_table,           # alias or table-literal -> table-literal
        "qualified_columns": qualified_columns,     # list of (qualifier, column)
        "select_unqualified": select_unqualified,   # list of unqualified columns from SELECT
    }

# ----------------- With SQLGLOT library ----------------
def extract_sql_references(expr):
    alias_to_table = {}
    qualified_columns = []
    select_unqualified = []

    def walk(node):
        if isinstance(node, exp.Table):
            alias = node.alias_or_name
            alias_to_table[alias.lower()] = node.name

        elif isinstance(node, exp.Column):
            if node.table:
                qualified_columns.append((node.table, node.name))
            else:
                select_unqualified.append(node.name)

        elif isinstance(node, exp.CTE):
            walk(node.this)

        elif isinstance(node, exp.Subquery):
            walk(node.this)

        elif isinstance(node, exp.Select):
            for child in node.expressions:
                walk(child)
            for clause in ["from", "joins", "where", "group", "having", "order"]:
                if clause in node.args:
                    val = node.args[clause]
                    if isinstance(val, list):
                        for item in val:
                            walk(item)
                    else:
                        walk(val)

        elif isinstance(node, exp.Expression):
            for child in node.args.values():
                if isinstance(child, list):
                    for item in child:
                        walk(item)
                elif isinstance(child, exp.Expression):
                    walk(child)

    walk(expr)
    return {
        "alias_to_table": alias_to_table,
        "qualified_columns": qualified_columns,
        "select_unqualified": select_unqualified
    }

# -----------------------
# Validator using KG
# -----------------------
from typing import Tuple, List
import sqlglot
from sqlglot import parse_one, exp
from sqlglot.expressions import Column, Alias

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


# def validate_sql_with_kg(sql_query: str, kg, max_suggest: int = 3) -> Tuple[bool, List[str]]:
#     """
#     Validate a SQL query against the Knowledge Graph (KG).
#     """
#     hints = []
#     valid = True

#     try:
#         parsed = parse_one(sql_query, read="sqlite")
#     except Exception as e:
#         return False, [f"❌ SQL parsing failed: {e}"]

#     # --- Extract CTE names ---
#     cte_names = {
#         cte.alias_or_name.lower()
#         for with_expr in parsed.find_all(exp.With)
#         for cte in with_expr.expressions
#         if cte.alias_or_name
#     }

#     # --- Map aliases to tables or CTEs ---
#     alias_to_table = {}

#     # Regular table aliases
#     for table_expr in parsed.find_all(exp.Table):
#         alias = table_expr.alias_or_name
#         table_name = table_expr.name
#         if alias:
#             alias_to_table[alias.lower()] = table_name.lower()
#         else:
#             # even without alias, register canonical
#             alias_to_table[table_name.lower()] = table_name.lower()

#     # Aliases for FROM/JOIN (covering CTEs aliased at runtime)
#     for from_or_join in list(parsed.find_all(exp.From)) + list(parsed.find_all(exp.Join)):
#         for source in from_or_join.expressions:
#             alias = source.alias_or_name
#             if alias:
#                 if isinstance(source.this, exp.Identifier):
#                     target_name = source.this.name
#                 elif isinstance(source.this, exp.Table):
#                     target_name = source.this.name
#                 else:
#                     target_name = None

#                 if target_name:
#                     # If it’s a CTE, map alias -> CTE name
#                     if target_name.lower() in cte_names:
#                         alias_to_table[alias.lower()] = target_name.lower()
#                     else:
#                         alias_to_table[alias.lower()] = target_name.lower()

#     # --- Prepare KG maps ---
#     table_map = get_table_names_from_kg(kg)
#     cols_by_table = get_columns_for_table_from_kg(kg)
#     canonical_tables = set(table_map.values())

#     # --- Extract columns ---
#     qualified_columns = []
#     unqualified_columns = []
#     for col in parsed.find_all(exp.Column):
#         if col.name == "*":
#             continue
#         if col.table:
#             qualified_columns.append((col.table.lower(), col.name))
#         else:
#             unqualified_columns.append(col.name)

#     # --- Validate qualified columns ---
#     for alias, col in qualified_columns:
#         col_l = col.lower()

#         if alias in alias_to_table:
#             resolved_table = canonical_table_name(kg, alias_to_table[alias]) or alias_to_table[alias]
#         elif alias in cte_names:
#             resolved_table = alias  # treat as valid CTE
#         else:
#             resolved_table = canonical_table_name(kg, alias) or alias

#         if resolved_table in cols_by_table:
#             if col_l not in cols_by_table[resolved_table]:
#                 candidates = [t for t, cols in cols_by_table.items() if col_l in cols]
#                 if candidates:
#                     suggestions = []
#                     for cand in candidates[:max_suggest]:
#                         path = find_join_path(kg, resolved_table, cand)
#                         suggestions.append(f"{cand} (join: {path['path_str'] if path else 'no direct path'})")
#                     hints.append(
#                         f"Column '{col}' not found in table '{resolved_table}'. "
#                         f"Found in: {', '.join(candidates[:max_suggest])}. "
#                         f"Suggestions: {', '.join(suggestions)}"
#                     )
#                 else:
#                     hints.append(
#                         f"Column '{col}' not found in table '{resolved_table}', "
#                         f"and no other table contains it."
#                     )
#                 valid = False
#         elif resolved_table in cte_names:
#             # ✅ allow CTE aliases without checking against KG
#             continue
#         else:
#             hints.append(
#                 f"Referenced table/alias '{alias}' could not be resolved. "
#                 f"Known tables: {sorted(canonical_tables)} + CTEs: {sorted(cte_names)}"
#             )
#             valid = False

#     # --- Validate unqualified columns ---
#     source_tables = set(alias_to_table.values())
#     source_canonicals = [canonical_table_name(kg, t) or t for t in source_tables] or list(cols_by_table.keys())

#     for col in unqualified_columns:
#         col_l = col.lower()
#         matches = [t for t in source_canonicals if col_l in cols_by_table.get(t, set())]

#         if len(matches) == 1:
#             continue
#         elif len(matches) > 1:
#             hints.append(f"Unqualified column '{col}' is ambiguous among tables {matches}. Please qualify it.")
#             valid = False
#         else:
#             found_in = [t for t, cols in cols_by_table.items() if col_l in cols]
#             if found_in:
#                 suggestions = []
#                 for cand in found_in[:max_suggest]:
#                     start = source_canonicals[0] if source_canonicals else cand
#                     path = find_join_path(kg, start, cand)
#                     suggestions.append(f"{cand} (path: {path['path_str'] if path else 'no path'})")
#                 hints.append(
#                     f"Column '{col}' not found in source tables {source_canonicals}. "
#                     f"Found in: {found_in[:max_suggest]}. "
#                     f"Suggested joins: {suggestions}"
#                 )
#             else:
#                 hints.append(f"Column '{col}' not found in any table.")
#             valid = False

#     return valid, hints


# def validate_sql_with_kg(sql_query: str, kg, max_suggest=3) -> Tuple[bool, List[str]]:
    """
    Validate SQL using KG.
    Returns (is_valid, hints_list)
    Hints describe missing tables/columns and suggested join paths.
    """
    parsed = parse_sql_references(sql_query)
    alias_to_table = parsed["alias_to_table"]  # e.g. {'p':'People', 'orders':'Orders'}
    qualified_columns = parsed["qualified_columns"]
    select_unqualified = parsed["select_unqualified"]

    # prepare KG canonical maps
    table_map = get_table_names_from_kg(kg)         # lower -> canonical
    cols_by_table = get_columns_for_table_from_kg(kg)  # canonical -> set(lower cols)
    canonical_tables = set(table_map.values())

    hints = []
    valid = True

    # --- Check qualified columns (alias.col or Table.col) ---
    for qual, col in qualified_columns:
        qual_l = qual.lower()
        col_l = col.lower()

        # resolve qual -> canonical table name
        resolved_table = None
        if qual_l in alias_to_table:
            # alias exists: alias_to_table[qual_l] is table literal used in SQL (maybe need canonical mapping)
            tbl_literal = alias_to_table[qual_l]
            # map literal to canonical (case-insensitive)
            resolved_table = canonical_table_name(kg, tbl_literal) or tbl_literal
        else:
            # maybe qual is table literal itself
            resolved_table = canonical_table_name(kg, qual) or qual

        # final canonical if possible
        if resolved_table and resolved_table in cols_by_table:
            if col_l not in cols_by_table[resolved_table]:
                # column missing from this table -> find candidate tables where column exists
                candidates = [t for t, cols in cols_by_table.items() if col_l in cols]
                if candidates:
                    # suggest top candidates and join paths
                    cand_strs = []
                    for cand in candidates[:max_suggest]:
                        path = find_join_path(kg, resolved_table, cand)  # returns dict with path_str
                        cand_strs.append(f"{cand} (join: {path['path_str'] if path else 'no direct path found'})")
                    hints.append(f"Column '{col}' not found in table '{resolved_table}'. Column exists in: {', '.join(candidates[:max_suggest])}. Suggestions: {', '.join(cand_strs)}")
                else:
                    hints.append(f"Column '{col}' not found in table '{resolved_table}', and no other table contains a column named '{col}'.")
                valid = False
        else:
            # resolved_table unknown
            hints.append(f"Referenced table/alias '{qual}' could not be resolved to a known table. Known tables: {sorted(canonical_tables)}")
            valid = False

    # --- Check unqualified SELECT columns (try to resolve using FROM tables) ---
    if select_unqualified:
        # build set of tables present in FROM/JOIN via alias mapping
        from_tables = set(alias_to_table.values())  # table literals as used in SQL
        # map to canonical
        from_canonical = []
        for t in from_tables:
            canon = canonical_table_name(kg, t)
            if canon:
                from_canonical.append(canon)
        if not from_canonical:
            # if no explicit FROM detection, use all KG tables as fallback
            from_canonical = list(cols_by_table.keys())

        for col in select_unqualified:
            col_l = col.lower()
            possible_tables = [t for t in from_canonical if col_l in cols_by_table.get(t, set())]
            if len(possible_tables) == 1:
                # good: resolved uniquely
                pass
            elif len(possible_tables) > 1:
                hints.append(f"Unqualified column '{col}' is ambiguous among tables {possible_tables}. Please qualify it (e.g., Table.Column).")
                valid = False
            else:
                # not found in immediate from-tables: search entire KG for where that column exists and suggest join path
                found_in = [t for t, cols in cols_by_table.items() if col_l in cols]
                if found_in:
                    paths = []
                    for cand in found_in[:max_suggest]:
                        # pick one of the from_canonical tables as starting if exists
                        start = from_canonical[0] if from_canonical else cand
                        path = find_join_path(kg, start, cand)
                        paths.append(f"{cand} (path: {path['path_str'] if path else 'no path'})")
                    hints.append(f"Column '{col}' not found in source tables {from_canonical}. It exists in {found_in[:max_suggest]}. Suggested joins: {paths}")
                else:
                    hints.append(f"Column '{col}' not found in any table in schema.")
                valid = False
    return valid, hints

# ----------------- JSON PARSER ----------------------
def safe_json_parse(plan_json):
    # Remove leading 'json' or markdown artifacts
    cleaned = re.sub(r"^json\s*", "", plan_json.strip(), flags=re.IGNORECASE)

    # Remove triple backticks if present
    if "```" in cleaned:
        cleaned = cleaned.split("```")[1].strip()

    return json.loads(cleaned)


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



# def answer_query_with_validation(nl_query, vectorstore, kg, max_retries=2):
#     """
#     Full pipeline:
#       1. RAG retrieval -> semantic_context
#       2. LLM generates SQL (first-pass)
#       3. validate_sql_with_kg -> hints if invalid
#       4. If invalid -> re-prompt LLM with structured hints + exact table names and KG paths
#       5. Run final validated SQL
#     """
#     # 1. retrieve semantic context
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
#     retrieved_docs = retriever.invoke(nl_query)
#     semantic_context = "\n".join([doc.page_content for doc in retrieved_docs])

#     # prepare canonical table list & relationship list for prompt
#     table_map = get_table_names_from_kg(kg)
#     canonical_tables = list(table_map.values())
#     # relationships extract for prompt readable form
#     rels = []
#     for u, v, data in kg.edges(data=True):
#         # show only table->table relation labels
#         if isinstance(u, tuple) and isinstance(v, tuple) and u[0]=="table" and v[0]=="table":
#             rels.append(f"{u[1]} -> {v[1]} ({data.get('relation')})")

#     # 2. LLM first pass
#     system_prompt = (
#         f"You are an expert SQL generator for SQLite.\n"
#         f"Use ONLY these exact table names (do NOT singularize or invent names): {canonical_tables}\n"
#         "Do NOT generate INSERT, UPDATE, DELETE, DROP, TRUNCATE.\n"
#         "Take care of values in the table irrespective user asked in capital or in lower case.\n"
#         "Return ONLY the final SELECT SQL (no explanation, no backticks).\n"
#         "Use provided schema context and join hints. Ensure the SQL is valid SQLite.\n"
#     )

#     user_prompt = f"""
#         Natural language query:
#         {nl_query}

#         Schema (retrieved context):
#         {semantic_context}

#         Known relationships:
#         {rels}
#     """

#     sql_query = llm_complete(system_prompt, user_prompt)  # your existing wrapper for LLM call
#     print("\n[LLM first-pass SQL]\n", sql_query)

#     # 3. Validate with KG and retry if needed
#     retries = 0
#     while retries < max_retries:
#         is_valid, hints = validate_sql_with_kg(sql_query, kg)
#         if is_valid:
#             break

#         # Build correction prompt with clear structured hints
#         hint_text = "\n".join(hints)
#         correction_prompt = f"""
#             The previously generated SQL is invalid for the schema. Issues:
#             {hint_text}

#             Use exact table names: {canonical_tables}
#             Known table relationships: {rels}

#             Please regenerate a corrected SELECT-only SQL query for: {nl_query}
#             Return only the SQL (no explanation).
#         """
#         print("\n[KG Hints]\n", hint_text)
#         sql_query = llm_complete(system_prompt, correction_prompt)
#         print(f"\n[LLM corrected SQL attempt {retries+1}]\n", sql_query)
#         retries += 1

#     # After retry loop, final validation
#     is_valid, hints = validate_sql_with_kg(sql_query, kg)
#     if not is_valid:
#         # give up with helpful message (do NOT execute)
#         raise ValueError(f"Unable to produce valid SQL after {max_retries} attempts. Hints: {hints}")

#     # 4. Execute final SQL
#     df = run_sql_query(sql_query)    # your existing db execution function
#     print(f"Sql Query: \n{sql_query}")
#     print(f"\n\n Output: \n{df}")
#     return sql_query, df



# ------------------- MAIN --------------------
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
    # nl_query = "Detect outlier orders: orders where the amount is greater than mean + 2*stddev for that Region."
    # nl_query = "List People who never returned an order but belong to Regions where return rate > 30%."
    nl_query = "Find the month with the highest return rate across all orders"
    answer_query_with_validation(nl_query, vectorstore, G)