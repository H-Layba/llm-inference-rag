import streamlit as st
from src.rag_pipeline import answer_query

st.set_page_config(
    page_title="Efficient LLM Inference RAG",
    page_icon="📚",
    layout="centered",
)

st.markdown(
    """
    <style>
    .main > div { padding-top: 2rem; }
    .subtitle { color: #9CA3AF; font-size: 1.05rem; line-height: 1.6; margin-bottom: 1.5rem; }
    .badge-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
    .badge {
        background: #1A1D27; border: 1px solid #2D3142; border-radius: 999px;
        padding: 0.3rem 0.9rem; font-size: 0.8rem; color: #A5B4FC;
    }
    .example-chip {
        display: inline-block; background: #1A1D27; border: 1px solid #2D3142;
        border-radius: 8px; padding: 0.5rem 0.9rem; margin: 0.25rem 0.4rem 0.25rem 0;
        font-size: 0.85rem; color: #C4B5FD; cursor: default;
    }
    .source-card {
        background: #1A1D27; border-left: 3px solid #818CF8; border-radius: 6px;
        padding: 0.7rem 1rem; margin-bottom: 0.5rem; font-size: 0.9rem;
    }
    .disclaimer {
        color: #6B7280; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #2D3142;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 📚 Efficient LLM Inference — Research Assistant")
st.markdown(
    '<div class="subtitle">A retrieval-augmented Q&A system over ~30 arXiv papers on '
    "quantization, KV-cache optimization, and speculative decoding. "
    "Answers are grounded in retrieved excerpts — every answer cites its sources.</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="badge-row">
        <span class="badge">🔍 Retrieval-Augmented</span>
        <span class="badge">📄 33 arXiv papers</span>
        <span class="badge">🧠 Mistral / Qwen via HF Inference</span>
        <span class="badge">⚡ FAISS + sentence-transformers</span>
    </div>
    """,
    unsafe_allow_html=True,
)

EXAMPLE_QUESTIONS = [
    "What is the difference between GPTQ and AWQ?",
    "How does speculative decoding speed up inference?",
    "What techniques exist for compressing the KV cache?",
    "What is PagedAttention and why does it help serving throughput?",
]

st.markdown("**Try asking:**")
chips_html = "".join(f'<span class="example-chip">{q}</span>' for q in EXAMPLE_QUESTIONS)
st.markdown(chips_html, unsafe_allow_html=True)
st.write("")

with st.form("query_form"):
    query = st.text_area(
        "Your question",
        placeholder="e.g. How does AWQ decide which weights to protect from quantization?",
        height=90,
    )
    col1, col2 = st.columns([3, 1])
    with col1:
        k = st.slider("Chunks to retrieve", min_value=2, max_value=8, value=5)
    with col2:
        st.write("")
        st.write("")
        submitted = st.form_submit_button("Ask →", type="primary", use_container_width=True)

if submitted:
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving relevant passages and generating an answer..."):
            try:
                answer, sources = answer_query(query, k=k)
                st.markdown("### Answer")
                st.markdown(answer)

                st.markdown("### Sources")
                for title, arxiv_id in sources:
                    st.markdown(
                        f'<div class="source-card">📄 <b>{title}</b><br>'
                        f'<span style="color:#9CA3AF;">arXiv:{arxiv_id}</span></div>',
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown(
    '<div class="disclaimer">Portfolio/demo project — answers are grounded in a fixed '
    "33-paper corpus and may not reflect the latest research. Always verify against the "
    "original papers for anything you rely on.</div>",
    unsafe_allow_html=True,
)