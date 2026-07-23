"""
src/ui.py
─────────
CRT terminal-themed Streamlit UI for QwenRAG.
Green phosphor on black. Scanlines. The whole deal.
"""

import os
import html
import streamlit as st

import config
from .rag_pipeline import RAGPipeline

# HTML escaping
#
# Everything below is rendered with unsafe_allow_html=True, so any
# document-derived or user-typed string interpolated into it is a script
# injection vector: a PDF whose text (or filename) contains
# <img src=x onerror=...> would execute JS when its chunk is shown in the
# sources panel. _esc() is the boundary between untrusted text and that HTML.
# The render-building is pulled into pure functions so it can be unit-tested
# without a Streamlit runtime (see tests/test_ui_escaping.py).

def _esc(s) -> str:
    """Escape untrusted text bound for an unsafe_allow_html sink (& < > \" ')."""
    return html.escape(str(s), quote=True)


def _user_bubble_html(content: str) -> str:
    return (
        '<div class="msg-user">'
        '<div class="msg-role-label">// INPUT</div>'
        f'<div class="bubble-user">{_esc(content)}</div>'
        '</div>'
    )


def _assistant_bubble_html(content: str) -> str:
    # Escape first, THEN turn newlines into <br> — so the line breaks are real
    # formatting while the content itself can never inject markup.
    safe = _esc(content).replace("\n", "<br>")
    return (
        '<div class="msg-assistant">'
        '<div class="msg-role-label">// OUTPUT</div>'
        f'<div class="bubble-assistant">{safe}</div>'
        '</div>'
    )


def _source_block_html(i: int, src: dict) -> str:
    page = f" | PG.{int(src['page_number'])}" if src.get("page_number") else ""
    text = src["text"]
    preview = text[:400] + ("..." if len(text) > 400 else "")
    return (
        '<div class="source-crt">'
        f'<div class="source-crt-header">REF {int(i)}: {_esc(src["source_file"])}{page}</div>'
        f'{_esc(preview)}'
        '</div>'
    )


def _file_row_html(finfo: dict) -> str:
    dot = "■" if finfo["is_indexed"] else "□"
    return (
        '<div class="file-crt">'
        f'<span class="file-crt-name">{_esc(finfo["name"])}</span>'
        f'<span class="file-crt-size">{dot} {_fmt_bytes(finfo["size_bytes"])}</span>'
        '</div>'
    )

# Session state

def _init_state():
    if "pipeline" not in st.session_state:
        # No args: every default now comes from config.py (and its env overrides).
        pipeline = RAGPipeline()
        pipeline.load()
        st.session_state["pipeline"] = pipeline
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = config.OLLAMA_MODEL
    if "top_k" not in st.session_state:
        st.session_state["top_k"] = config.TOP_K

def _get_pipeline() -> RAGPipeline:
    return st.session_state["pipeline"]

def _fmt_bytes(n: int) -> str:
    if n < 1024:       return f"{n}B"
    if n < 1024 ** 2:  return f"{n/1024:.0f}KB"
    return f"{n/1024**2:.1f}MB"

# Sidebar

def _render_sidebar():
    pipeline = _get_pipeline()

    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 18px 0; border-bottom:1px solid #1a4d1a; margin-bottom:14px;">
          <div style="font-family:'VT323',monospace; font-size:1.8rem; color:#33ff33;
                      text-shadow:0 0 10px #00ff00; letter-spacing:0.08em;">
            RAG v1.0
          </div>
          <div style="font-family:'Share Tech Mono',monospace; font-size:0.68rem;
                      color:#1a7a1a; letter-spacing:0.18em; margin-top:2px;">
            LOCAL DOCUMENT RETRIEVAL SYSTEM
          </div>
        </div>
        """, unsafe_allow_html=True)

    
        st.markdown('<div class="crt-label">SYSTEM STATUS</div>', unsafe_allow_html=True)
        status = pipeline.get_llm_status()

        if status["running"]:
            st.markdown('<span class="crt-pill crt-pill-ok">[OK] OLLAMA ONLINE</span>', unsafe_allow_html=True)
            if status["model_ready"]:
                st.markdown(f'<span class="crt-pill crt-pill-ok">[OK] MODEL LOADED</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="crt-pill crt-pill-warn">[!!] MODEL NOT FOUND</span>', unsafe_allow_html=True)
                st.caption(f">> ollama pull {status['model']}")
        else:
            st.markdown('<span class="crt-pill crt-pill-err">[ERR] OLLAMA OFFLINE</span>', unsafe_allow_html=True)
            st.caption(">> ollama serve")

        # EMBEDDING_MODEL was changed, but the index on disk was built with a
        # different encoder. We are still querying with the OLD one — say so,
        # rather than letting config look like it took effect.
        if getattr(pipeline.vector_store, "model_mismatch", False):
            st.markdown(
                '<span class="crt-pill crt-pill-warn">[!!] EMBED MODEL MISMATCH</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                f">> index built with '{pipeline.vector_store.model_name}', "
                f"config wants '{config.EMBEDDING_MODEL}' — click >> REINDEX"
            )

        st.write("")

        st.markdown('<div class="crt-label">SELECT MODEL</div>', unsafe_allow_html=True)
        available = status.get("models", [])
        defaults = [config.OLLAMA_MODEL, "qwen2.5:3b", "qwen2.5:14b", "llama3.2:3b", "mistral:7b"]
        options = list(dict.fromkeys(available + defaults))
        cur = st.session_state["llm_model"]
        if cur not in options:
            options.insert(0, cur)

        sel = st.selectbox("model", options=options, index=options.index(cur), label_visibility="collapsed")
        if sel != cur:
            st.session_state["llm_model"] = sel
            pipeline.llm.model = sel
            st.toast(f"MODEL >> {sel}", icon="🖥")

        st.markdown('<div class="crt-label">RETRIEVAL DEPTH</div>', unsafe_allow_html=True)
        top_k = st.slider(
            "chunks", 1, config.UI_MAX_TOP_K, st.session_state["top_k"],
            label_visibility="collapsed",
        )
        st.session_state["top_k"] = top_k
        pipeline.top_k = top_k

        st.markdown('<div class="crt-label">LOAD DOCUMENTS</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "drop files",
            type=["pdf","txt"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded:
            if st.button(">> INDEX FILES", use_container_width=True):
                _handle_uploads(uploaded, pipeline)
                st.rerun()

        file_list = pipeline.get_uploaded_files()
        if file_list:
            st.markdown(
                f'<div class="crt-label">LOADED FILES [{len(file_list)}]</div>',
                unsafe_allow_html=True,
            )
            for finfo in file_list:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(_file_row_html(finfo), unsafe_allow_html=True)
                with col2:
                    if st.button("DEL", key=f"del_{finfo['name']}",
                                 help=f"Delete {finfo['name']}", use_container_width=True):
                        pipeline.delete_file(finfo["name"])
                        st.toast(f"DELETED: {finfo['name']}")
                        st.rerun()

        st.write("")
        # Full-width, stacked — side-by-side columns were too narrow and clipped
        # the labels against the sidebar edge.
        if st.button(">> REINDEX", use_container_width=True):
            with st.spinner("REINDEXING..."):
                added, _ = pipeline.reindex_all()
            st.toast(f"INDEXED {added} CHUNKS")
            st.rerun()
        if st.button(">> CLEAR LOG", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

        st.write("")
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.68rem; '
            f'color:#1a4d1a; letter-spacing:0.08em;">'
            f'MEM CHUNKS: {pipeline.chunk_count()} | '
            f'FILES: {len(pipeline.get_indexed_files())}'
            f'</div>',
            unsafe_allow_html=True,
        )

# Upload handler

def _handle_uploads(uploaded_files, pipeline: RAGPipeline):
    saved = []
    prog = st.progress(0, text="SAVING FILES...")
    for i, f in enumerate(uploaded_files):
        path = pipeline.save_uploaded_file(f, f.name)
        saved.append(path)
        prog.progress((i+1)/len(uploaded_files), text=f"SAVED: {f.name}")

    prog.progress(0, text="GENERATING EMBEDDINGS...")

    def _cb(cur, tot, fname):
        if tot:
            prog.progress(cur/tot, text=f"PROCESSING [{cur}/{tot}]: {fname}")

    added, errors = pipeline.ingest_files(saved, progress_callback=_cb)
    prog.empty()
    if added:
        st.toast(f"OK: {added} CHUNKS FROM {len(saved)} FILE(S)")
    for err in errors:
        st.error(err)

# Chat rendering

# Sample queries shown in the empty state: (compact button label, full query sent).
_SAMPLE_QUERIES = [
    ("▸ SUMMARISE",       "Summarise the key points of this document."),
    ("▸ KEY FINDINGS",    "What are the main conclusions or findings?"),
    ("▸ NUMBERS & DATES", "List any important numbers or dates mentioned."),
    ("▸ MAIN PROBLEM",    "What problem does this document address?"),
]


def _render_empty_state(pipeline: RAGPipeline):
    """Centered middle-of-screen state: doc hints, or compact sample-query boxes."""
    file_list = pipeline.get_uploaded_files()

    if not file_list:
        st.markdown(
            '<div class="crt-empty" style="padding:12vh 20px 24px;">'
            '<span class="crt-empty-art">█▓▒░</span>'
            '<div class="crt-empty-text">AWAITING DOCUMENTS<span class="blink">_</span><br><br>'
            '<span style="color:#1a4d1a; font-size:0.72rem;">'
            'LOAD PDF / TXT VIA THE SIDEBAR &gt; THEN ASK BELOW</span></div></div>',
            unsafe_allow_html=True,
        )
        return

    if pipeline.chunk_count() == 0:
        st.markdown(
            '<div class="crt-empty" style="padding:12vh 20px 24px;">'
            '<span class="crt-empty-art" style="color:#7a5500;">▓▒░</span>'
            '<div class="crt-empty-text" style="color:#7a5500;">FILES DETECTED — INDEX EMPTY'
            '<span class="blink">_</span><br><br><span style="font-size:0.72rem;">'
            'CLICK &gt;&gt; INDEX FILES IN THE SIDEBAR</span></div></div>',
            unsafe_allow_html=True,
        )
        return

    # Indexed and ready — compact, centered sample-query boxes.
    st.markdown(
        '<div style="text-align:center; padding:9vh 0 14px;">'
        '<span class="crt-empty-art" style="font-size:3rem;">▚▞▚</span>'
        '<div class="crt-empty-text" style="margin-top:6px;">READY — PICK A SAMPLE OR TYPE BELOW'
        '<span class="blink">_</span></div></div>',
        unsafe_allow_html=True,
    )
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        grid = st.columns(2)
        for i, (label, query) in enumerate(_SAMPLE_QUERIES):
            with grid[i % 2]:
                if st.button(label, key=f"ex_{i}", use_container_width=True):
                    _handle_query(query, pipeline)
                    st.rerun()


def _render_chat(pipeline: RAGPipeline):
    history = st.session_state["chat_history"]

    if not history:
        _render_empty_state(pipeline)
        return

    for msg in history:
        if msg["role"] == "user":
            st.markdown(_user_bubble_html(msg["content"]), unsafe_allow_html=True)
        else:
            st.markdown(_assistant_bubble_html(msg["content"]), unsafe_allow_html=True)

            if msg.get("no_retrieval"):
                st.markdown(
                    '<span class="crt-pill crt-pill-warn">'
                    '[!!] NO RETRIEVAL &mdash; NOT GROUNDED IN YOUR DOCUMENTS'
                    '</span>',
                    unsafe_allow_html=True,
                )

            # The context window filled up and the lowest-ranked chunks were cut.
            # Say so — a silently shortened context is what caused this bug class.
            if msg.get("budget_dropped"):
                st.markdown(
                    f'<span class="crt-pill crt-pill-warn">'
                    f'[!!] CONTEXT BUDGET &mdash; USED {msg["budget_total"] - msg["budget_dropped"]}'
                    f'/{msg["budget_total"]} CHUNKS ({msg["budget_dropped"]} LOWEST-RANKED DROPPED)'
                    f'</span>',
                    unsafe_allow_html=True,
                )

            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"[SRC] {len(sources)} REFERENCE CHUNK(S)", expanded=False):
                    for i, src in enumerate(sources, 1):
                        st.markdown(_source_block_html(i, src), unsafe_allow_html=True)

# Query handler

def _handle_query(question: str, pipeline: RAGPipeline):
    st.session_state["chat_history"].append({
        "role": "user", "content": question, "sources": [],
    })

    status = pipeline.get_llm_status()
    if not status["running"]:
        ans = "ERR: OLLAMA NOT RUNNING. EXECUTE >> ollama serve"
        st.session_state["chat_history"].append({"role":"assistant","content":ans,"sources":[]})
        return
    if not status["model_ready"]:
        ans = f"ERR: MODEL '{status['model']}' NOT LOADED.\nEXECUTE >> ollama pull {status['model']}"
        st.session_state["chat_history"].append({"role":"assistant","content":ans,"sources":[]})
        return

    with st.spinner("QUERYING VECTOR INDEX... GENERATING RESPONSE..."):
        history_for_llm = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["chat_history"][:-1]
        ]
        stream = pipeline.query(question=question, chat_history=history_for_llm, stream=True)
        full = "".join(stream)

    sources = pipeline.get_last_sources()
    build = pipeline.last_prompt_build()
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": full,
        "sources": sources,
        "no_retrieval": pipeline.last_was_no_retrieval(),
        "budget_dropped": len(build.dropped_chunks) if build else 0,
        "budget_total": (len(build.used_chunks) + len(build.dropped_chunks)) if build else 0,
    })

# Main render

def render_app():
    _init_state()
    pipeline = _get_pipeline()

    _render_sidebar()

    st.markdown("""
    <div class="crt-header">
      <div class="crt-title">INTELLIGENCE TERMINAL</div>
      <div class="crt-subtitle">
        LOCAL RAG SYSTEM &nbsp;|&nbsp; QWEN 2.5 &nbsp;|&nbsp; FAISS VECTOR SEARCH
        &nbsp;<span class="cursor">_</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Messages (or the centered empty state) fill the space between the header and
    # the input, which st.chat_input pins to the bottom of the screen.
    _render_chat(pipeline)

    prompt = st.chat_input("C:\\QUERY>  type a question and press ENTER")
    if prompt and prompt.strip():
        _handle_query(prompt.strip(), pipeline)
        st.rerun()
