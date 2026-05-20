# ============================================
# IMPORTS
# ============================================

import streamlit as st
import requests


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(

    page_title="Advanced RAG Assistant",

    page_icon="🤖",

    layout="wide"
)


# ============================================
# CUSTOM CSS
# ============================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 2rem;
    }

    .stChatMessage {
        padding: 1rem;
        border-radius: 12px;
    }

    .source-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================
# TITLE
# ============================================

st.title("🤖 Advanced RAG Assistant")

st.markdown(
    """
Ask questions grounded in your enterprise documents.

Powered by:
- Hybrid Retrieval
- Weighted RRF
- Cross-Encoder Reranking
- Groq + Gemini Fallback
- ChromaDB
"""
)


# ============================================
# API CONFIG
# ============================================

API_URL = (
    "http://127.0.0.1:8000/query-stream"
)


# ============================================
# SESSION STATE
# ============================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================
# DISPLAY CHAT HISTORY
# ============================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================
# CHAT INPUT
# ============================================

query = st.chat_input(
    "Ask a question..."
)


# ============================================
# HANDLE QUERY
# ============================================

if query:

    # ----------------------------------------
    # STORE USER MESSAGE
    # ----------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": query
    })

    # ----------------------------------------
    # DISPLAY USER MESSAGE
    # ----------------------------------------

    with st.chat_message("user"):

        st.markdown(query)

    # ----------------------------------------
    # ASSISTANT RESPONSE
    # ----------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        try:

            with st.spinner(
                "Retrieving and generating answer..."
            ):

                response = requests.post(

                    API_URL,

                    json={

                        "query": query
                    },

                    stream=True,

                    timeout=300
                )

            # --------------------------------
            # STATUS CHECK
            # --------------------------------

            if response.status_code != 200:

                st.error(

                    f"API Error: "
                    f"{response.status_code}"
                )

            else:

                # ----------------------------
                # STREAM TOKENS
                # ----------------------------

                for chunk in response.iter_content(

                    chunk_size=1,

                    decode_unicode=True
                ):

                    if chunk:

                        full_response += chunk

                        response_placeholder.markdown(

                            full_response
                        )

                # ----------------------------
                # STORE CHAT HISTORY
                # ----------------------------

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": full_response
                })

        except Exception as e:

            st.error(str(e))