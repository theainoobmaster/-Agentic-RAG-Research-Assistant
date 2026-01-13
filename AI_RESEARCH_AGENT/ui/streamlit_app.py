import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="AI Research Agent", layout="wide")
st.title("🧠 Agentic AI Research Assistant")

# ---------------- Input ----------------
query = st.text_input(
    "Ask a question",
    placeholder="Explain how transformers work",
)

paper_mode = st.checkbox("📄 Ask about a research paper")

paper_url = None
if paper_mode:
    paper_url = st.text_input(
        "Paper URL (arXiv / PDF)",
        placeholder="https://arxiv.org/abs/1706.03762",
    )

# ---------------- Submit ----------------
if st.button("Run"):
    if not query:
        st.warning("Please enter a question.")
    else:
        payload = {"query": query}

        if paper_url:
            payload["paper_url"] = paper_url

        with st.spinner("Thinking..."):
            response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            with st.expander("Thoughts"):
                for t in data.get("thoughts", []):
                    st.write(t)

            with st.expander("Actions"):
                st.json(data.get("actions", []))
        else:
            st.error(f"Error: {response.text}")
