import streamlit as st
from pathlib import Path
from src.config import VECTORSTORE_DIR

from src.retrieve import (
    load_index,
    load_metadata,
    build_query_encoder,
    retrieve_with_metadata,
    generate_answer,
    _load_chunk_text_map,
)

st.set_page_config(
    page_title="Lumora v1.0",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>

.stChatMessage {
    border-radius: 18px;
    padding: 12px;
    margin-bottom: 10px;
}

[data-testid="stChatMessageContent"] {
    font-size: 16px;
    line-height: 1.7;
}

.stChatInput {
    border-radius: 16px;
}
            /* Lumora Branding */

:root {
    --accent: #7C3AED;
}

h1 {
    font-weight: 700;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

MIN_SCORE = 0.64

@st.cache_resource
def load_rag():
    index = load_index()
    metadata = load_metadata()

    text_map = _load_chunk_text_map(VECTORSTORE_DIR)

    if text_map:
        metadata = [
            {
                **row,
                "text": text_map.get(
                    str(row.get("chunk_id", "")),
                    str(row.get("text", ""))
                ),
            }
            for row in metadata
        ]

    model = build_query_encoder()
    return index, metadata, model


index, metadata, model = load_rag()

# Branding Header

# Lumora Header

st.markdown("""
<h1 style='
margin-bottom:0px;
font-size:42px;
font-weight:700;
'>
Lumora v1.0
</h1>

<p style='
color:#9CA3AF;
font-size:16px;
margin-top:0px;
margin-bottom:8px;
'>
Learn • Understand • Revise
</p>
""", unsafe_allow_html=True)
st.markdown("""
<div style='
display:flex;
gap:12px;
align-items:center;
margin-top:-4px;
margin-bottom:20px;
'>

<span style='
background:#7C3AED;
padding:4px 10px;
border-radius:999px;
font-size:13px;
font-weight:600;
color:white;
'>
v1.0
</span>

<span style='
color:#9CA3AF;
font-size:14px;
'>
Powered by RAG + Gemini + LM Studio
</span>

</div>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.header("Lumora Settings ⚙️")

top_k = st.sidebar.slider(
    "Top-k retrieval",
    1,
    5,
    5
)

show_debug = st.sidebar.checkbox(
    "Show retrieval details"
)

st.sidebar.markdown("---")
st.sidebar.caption("Generation Layer")
st.sidebar.success("Gemini + LM Studio Fallback (meta-llama-3.1-8b)")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg.get("diagram"):

            st.markdown("---")
            st.caption("🖼 Related Diagram")

            st.image(
                msg["diagram"],
                use_container_width=True
            )

# Chat input
question = st.chat_input(
    "Hey there, chat here!"
)

if question:

    # User message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Searching knowledge base..."):

            results = retrieve_with_metadata(
                query=question,
                index=index,
                metadata=metadata,
                model=model,
                top_k=top_k,
            )

        if not results:
            response = "❌ No matches found."

        elif results[0].score < MIN_SCORE:
            response = (
                f"❌ No relevant information found "
                f"(score={results[0].score:.4f})"
            )

        else:

            with st.spinner("Generating answer..."):
                response = generate_answer(question, results)

        # Answer
        st.markdown(response)

        # Diagram Layer
        diagram_path = None
        query_lower = question.lower()

        # Query-aware match
        for r in results:

            diagram = getattr(r, "diagram", None)

            if not diagram:
                continue

            filename = Path(diagram).stem.lower()

            if any(word in filename for word in query_lower.split()):

                candidate = Path("data/diagrams") / diagram

                if candidate.exists():
                    diagram_path = candidate
                    break

        # Fallback
        if not diagram_path:

            for r in results:

                diagram = getattr(r, "diagram", None)

                if diagram:

                    candidate = Path("data/diagrams") / diagram

                    if candidate.exists():
                        diagram_path = candidate
                        break

        # Display
        if diagram_path:

            st.markdown("---")
            st.caption("🖼 Related Diagram")

            st.image(
                diagram_path.as_posix(),
                use_container_width=True
            )

    # Save assistant message + diagram
    st.session_state.messages.append(
    {
        "role": "assistant",
        "content": response,
        "diagram": (
            diagram_path.as_posix()
            if diagram_path else None
        )
    }
)
