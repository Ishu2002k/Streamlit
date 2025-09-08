import streamlit as st
import os
from dotenv import load_dotenv

from admin import admin_panel
from user import user_panel
from visualize import visualize_panel

# --- Load Environment Variables ---
# This ensures we consistently use the database connection string everywhere.
load_dotenv()
DATABASE_CONNECTION_STRING = os.getenv("DATABASE_CONNECTION_STRING")

# App Title and Branding
st.set_page_config(
    initial_sidebar_state="collapsed",
    page_title="GenAI SQL Assistant",
    page_icon="📊",
    layout="wide"
)

# --- Fix ghost strip when collapsed ---
hide_sidebar_style = """
    <style>
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0px !important;
            min-width: 0px !important;
            margin-left: -20px !important; /* removes leftover strip */
        }
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

# --- Page background styling ---
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #D4D4D4; /* Light green */
    padding-top: 0rem; /* Reduce top padding */
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0); /* Transparent header */
}
[data-testid="stSidebar"] {
    background-color: #D4D4D4; /* Optional: light green sidebar */
}
.block-container {
    padding-top: 0rem !important;  /* Remove Streamlit default top gap */
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# --- Branding Section ---
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="margin-top:-10px; background-color: transparent;">
        <img src="data:image/png;base64,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" width="140" style="background:transparent;">
    </div>
    """,
    unsafe_allow_html=True
)


# st.image(
#     "axtria_logo.png", 
#     width=140
# )
st.markdown(
    "<h2 style='text-align: center; color: #2E86C1; margin-top: -10px;'>📊 GenAI SQL Assistant</h2>",
    unsafe_allow_html=True
)
st.markdown("</div>", unsafe_allow_html=True)
st.write("---")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"  # default to role selection

# --- Home Page (Role Selection) ---
if st.session_state.page == "home":
    st.markdown("### 🔑 Please select your role")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("👨‍💼 Admin", use_container_width=True):
            st.session_state.page = "admin"
            st.rerun()

    with col2:
        if st.button("👤 User", use_container_width=True):
            st.session_state.page = "user"
            st.rerun()
    
    with col3:
        if st.button("📊 Visualize Data", use_container_width=True):
            st.session_state.page = "visualize"
            st.rerun()

# --- Admin Page ---
elif st.session_state.page == "admin":
    st.sidebar.button("⬅️ Back to Home", on_click=lambda: st.session_state.update(page="home"))
    admin_panel()

# --- User Page ---
elif st.session_state.page == "user":
    st.sidebar.button("⬅️ Back to Home", on_click=lambda: st.session_state.update(page="home"))
    st.markdown("### 👤 User Dashboard")
    user_panel()

# --- Visualization Page ---
elif st.session_state.page == "visualize":
    st.sidebar.button("⬅️ Back to Home", on_click=lambda: st.session_state.update(page="home"))
    
    # --- IMPORTANT CHANGE ---
    # Instead of a hardcoded local file, we now pass the consistent
    # database connection string from your .env file.
    if DATABASE_CONNECTION_STRING:
        visualize_panel(DATABASE_CONNECTION_STRING)
    else:
        st.error("DATABASE_CONNECTION_STRING not found in .env file. Please configure it in your .env file.")