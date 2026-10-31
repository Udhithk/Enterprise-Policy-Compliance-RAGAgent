"""
app.py
Streamlit dashboard for the Enterprise Policy Compliance RAG Agent.
Run:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from rag_engine import EnterpriseRAGAgent, RAGConfig, RAGResponse

st.set_page_config(
    page_title="Policy Compliance RAG Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1100px;}
      .hero {padding: 1.1rem 1.4rem; border-radius: 14px; margin-bottom: 1rem;
             background: linear-gradient(135deg, #1e3a8a 0%, #4f46e5 100%); color: #fff;}
      .hero h1 {margin: 0; font-size: 1.6rem;}
      .hero p {margin: .25rem 0 0; opacity: .9; font-size: .95rem;}
      .badge {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
              font-weight: 600; margin-right: 6px;}
      .badge-cited {background:#dcfce7; color:#166534;}
      .badge-ctx {background:#e5e7eb; color:#374151;}
      .badge-table {background:#fef3c7; color:#92400e;}
      .badge-text {background:#dbeafe; color:#1e40af;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model and vector store...")
def get_agent() -> EnterpriseRAGAgent:
    return EnterpriseRAGAgent(RAGConfig())


def init_state() -> None:
    st.session_state.setdefault("messages", [])  # list of {"role", "content", "response"?}
    st.session_state.setdefault("uploader_key", 0)


init_state()
agent = get_agent()


# --------------------------------------------------------------------------- #
# Sidebar: API key, upload, indexed documents, settings
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Control Panel")

    key_in_env = bool(os.getenv("GOOGLE_API_KEY"))
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        placeholder="Using GOOGLE_API_KEY from environment" if key_in_env else "Paste your key",
        help="Leave blank to use the GOOGLE_API_KEY environment variable.",
    )
    if api_key:
        agent.set_api_key(api_key)
        os.environ["GOOGLE_API_KEY"] = api_key
        try:
            from pathlib import Path
            p = Path(".env")
            content = p.read_text(encoding="utf-8") if p.exists() else ""
            if "GOOGLE_API_KEY=" in content:
                lines = [f"GOOGLE_API_KEY={api_key}\n" if l.startswith("GOOGLE_API_KEY=") else l for l in content.splitlines(keepends=True)]
                p.write_text("".join(lines), encoding="utf-8")
            else:
                p.write_text(f"GOOGLE_API_KEY={api_key}\n" + content, encoding="utf-8")
        except Exception:
            pass

    model_options = [
        "gemini-3.8-flash",
        "gemini-3.8-pro",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]
    cur_model = agent.config.llm_model
    if cur_model not in model_options:
        model_options.insert(0, cur_model)
    default_idx = (
        model_options.index("gemini-3.8-flash")
        if cur_model == "gemini-2.5-flash"
        else model_options.index(cur_model)
    )
    selected_model = st.selectbox(
        "Gemini model",
        options=model_options,
        index=default_idx,
        help="Select which Gemini model to use for query rewriting and answer generation.",
    )
    agent.set_model(selected_model)

    st.divider()
    st.subheader("📄 Upload policy PDFs")
    uploads = st.file_uploader(
        "Drag and drop enterprise PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if st.button("Process & index", type="primary", use_container_width=True, disabled=not uploads):
        overall = st.progress(0.0, text="Starting...")
        with st.status("Ingesting documents...", expanded=True) as status:
            for i, f in enumerate(uploads):
                st.write(f"**{f.name}**")

                def _cb(frac: float, msg: str, _i=i, _n=len(uploads)) -> None:
                    overall.progress(min((_i + frac) / _n, 1.0), text=f"{f.name}: {msg}")

                try:
                    res = agent.ingest_pdf(f.getvalue(), f.name, progress=_cb)
                    if res.skipped:
                        st.write(f"↪ {res.message}")
                    else:
                        st.write(f"✅ {res.pages} pages, {res.tables} tables, {res.chunks} chunks")
                except Exception as exc:
                    st.write(f"❌ {exc}")
            overall.progress(1.0, text="All files processed.")
            status.update(label="Ingestion complete", state="complete", expanded=False)
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()
    st.subheader("🗂️ Indexed documents")
    docs = agent.list_documents()
    if not docs:
        st.caption("Nothing indexed yet.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d['source']}**  \n{d['pages']} pp · {d['chunks']} chunks · {d['tables']} tables")
        if c2.button("🗑️", key=f"del_{d['source']}", help="Remove from index"):
            agent.delete_document(d["source"])
            st.rerun()

    st.divider()
    st.subheader("🔎 Retrieval")
    agent.config.final_k = st.slider("Snippets sent to LLM", 2, 10, agent.config.final_k)
    st.caption(
        f"Hybrid: BM25 {agent.config.bm25_weight:.0%} + Dense {agent.config.dense_weight:.0%} "
        f"(RRF) · {agent.config.llm_model}"
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_evidence(resp: RAGResponse) -> None:
    if not resp.retrieved:
        return
    cited = set(resp.cited_indices)
    label = f"🔍 Retrieved evidence ({len(resp.retrieved)} snippets, {len(cited)} cited)"
    with st.expander(label):
        if resp.standalone_question:
            st.caption(f"Search query used: _{resp.standalone_question}_")
        for i, doc in enumerate(resp.retrieved, start=1):
            m = doc.metadata
            badge = (
                '<span class="badge badge-cited">CITED</span>'
                if i in cited
                else '<span class="badge badge-ctx">CONTEXT</span>'
            )
            kind = m.get("content_type", "text")
            kind_badge = f'<span class="badge badge-{kind}">{kind.upper()}</span>'
            with st.container(border=True):
                st.markdown(
                    f"{badge}{kind_badge}**[S{i}]** {m.get('source')} · "
                    f"page {m.get('page')} · _{m.get('section')}_",
                    unsafe_allow_html=True,
                )
                if kind == "table":
                    st.markdown(doc.page_content.split("\n", 1)[-1])
                else:
                    st.text(doc.page_content)


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛡️"):
        st.markdown(msg["content"])
        resp: RAGResponse | None = msg.get("response")
        if resp is not None:
            st.caption(
    f"⏱ {resp.latency_s:.1f}s total · retrieval {resp.retrieval_s:.2f}s · Gemini {resp.generation_s:.1f}s"
    + (" · refused (not in documents)" if resp.refused else "")
)


# --------------------------------------------------------------------------- #
# Main page
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <div class="hero">
      <h1>🛡️ Enterprise Policy Compliance Agent</h1>
      <p>Hybrid RAG (BM25 + vector search) over your policy PDFs. Every answer is cited
         or refused.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    if docs:
        st.info("Ask a question about your indexed policies, e.g. a clause, threshold, or policy ID.")
    else:
        st.info("Upload and index at least one PDF from the sidebar to begin.")

for message in st.session_state.messages:
    render_message(message)

question = st.chat_input("Ask about a policy, clause number, threshold, or table value...")
if question:
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    # (user, assistant) pairs for follow-up question rewriting
    msgs = st.session_state.messages[:-1]
    history = [
        (msgs[i]["content"], msgs[i + 1]["content"])
        for i in range(0, len(msgs) - 1, 2)
        if msgs[i]["role"] == "user" and msgs[i + 1]["role"] == "assistant"
    ]

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Retrieving (BM25 + dense) and verifying against sources..."):
            try:
                resp = agent.ask(question, history=history)
                assistant_msg = {"role": "assistant", "content": resp.answer, "response": resp}
            except Exception as exc:
                assistant_msg = {"role": "assistant", "content": f"⚠️ Error: {exc}", "response": None}
    st.session_state.messages.append(assistant_msg)
    st.rerun()
