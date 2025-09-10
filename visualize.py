# Visualization with SQLite Cloud support
import sqlite3
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlitecloud  # ✅ for SQLite Cloud


def visualize_panel(db_path="uploaded_db.sqlite", use_cloud=True):
    st.title("📊 Smart Table Visualization")

    # -------------------------
    # Connect to SQLite DB
    # -------------------------
    if use_cloud:
        # ✅ SQLite Cloud connection
        conn = sqlitecloud.connect(
            "sqlitecloud://cbwb6jhxhk.g1.sqlite.cloud:8860/user_info?apikey=tzKSY69TJgit4JxRZqGYxSSSXXn5EWfmoYezjolRdn8"
        )
    else:
        # ✅ Local SQLite connection
        conn = sqlite3.connect(db_path)

    # -------------------------
    # Get list of tables
    # -------------------------
    tables = []
    cursor = conn.cursor()

    if use_cloud:
        # ✅ SQLite Cloud uses pragma_table_list
        cursor.execute("PRAGMA table_list;")
        result = cursor.fetchall()
        if result:
            # result = (schema, name, type, ncol, wr, strict)
            tables = [
                r[1] for r in result
                if r[2] == "table"
                and not r[1].startswith("_sqliteai_")
                and not r[1].startswith("sqlite_")
            ]
    else:
        # ✅ Local SQLite
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' 
              AND name NOT LIKE 'sqlite_%'
              AND name NOT LIKE '_sqliteai_%';
        """)
        result = cursor.fetchall()
        tables = [r[0] for r in result] if result else []

    if not tables:
        st.warning("⚠️ No tables found in the database.")
        conn.close()
        return

    # -------------------------
    # Table selection
    # -------------------------
    table_name = st.selectbox("Select a table", tables)
    if not table_name:
        conn.close()
        return

    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()

    # -------------------------
    # Data Overview
    # -------------------------
    st.subheader("📋 Data Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write("**Missing Values:**")
    st.dataframe(
        df.isna().sum().reset_index().rename(columns={0: "Missing Values", "index": "Column"})
    )

    st.write("**Preview:**")
    st.dataframe(df.head(10))

    # Column types
    st.subheader("📑 Column Types")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str),
        "Unique Values": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(col_info)

    # -------------------------
    # Statistics
    # -------------------------
    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all"))

    # -------------------------
    # Visualizations
    # -------------------------
    st.subheader("📈 Visualizations")

    # Identify column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

    # Correlation heatmap
    if len(numeric_cols) >= 2:
        st.markdown("### 🔥 Correlation Heatmap")
        corr = df[numeric_cols].corr()

        # Mask upper triangle (including diagonal)
        mask = np.triu(np.ones_like(corr, dtype=bool))

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, cmap="coolwarm", mask=mask, ax=ax, linewidths=0.5)
        ax.set_title("Correlation Heatmap (without self-correlation)")
        st.pyplot(fig)

    # Univariate histograms
    if numeric_cols:
        st.markdown("### 📊 Distributions (Numeric)")
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color="skyblue")
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    # Categorical bar plots
    if categorical_cols:
        st.markdown("### 📦 Categorical Counts")
        for col in categorical_cols[:5]:  # limit to avoid clutter
            fig, ax = plt.subplots()
            df[col].value_counts().head(10).plot(kind="bar", ax=ax, color="orange")
            ax.set_title(f"Top 10 categories in {col}")
            st.pyplot(fig)

    # Numeric vs. Categorical (boxplot)
    if numeric_cols and categorical_cols:
        st.markdown("### 📊 Numeric vs. Categorical")
        x_col = st.selectbox("Select categorical column", categorical_cols)
        y_col = st.selectbox("Select numeric column", numeric_cols)

        fig, ax = plt.subplots()
        sns.boxplot(x=df[x_col], y=df[y_col], ax=ax)
        ax.set_title(f"{y_col} by {x_col}")
        st.pyplot(fig)

    # Numeric vs. Numeric (scatter)
    if len(numeric_cols) >= 2:
        st.markdown("### 🔍 Numeric vs Numeric")
        cols = st.multiselect("Select two numeric columns", numeric_cols, default=numeric_cols[:2])
        if len(cols) == 2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=cols[0], y=cols[1], ax=ax, alpha=0.7)
            ax.set_title(f"{cols[0]} vs {cols[1]}")
            st.pyplot(fig)

    # Time-series plots
    if datetime_cols and numeric_cols:
        st.markdown("### ⏳ Time Series")
        time_col = st.selectbox("Select datetime column", datetime_cols)
        value_col = st.selectbox("Select numeric column for trend", numeric_cols)

        df_sorted = df.dropna(subset=[time_col]).sort_values(time_col)
        fig, ax = plt.subplots()
        sns.lineplot(x=df_sorted[time_col], y=df_sorted[value_col], ax=ax)
        ax.set_title(f"{value_col} over time ({time_col})")
        st.pyplot(fig)




