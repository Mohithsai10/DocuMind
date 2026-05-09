"""
app.py — Gradio 6.x web interface for DocuMind.

Two-column layout: document ingestion on the left, chat on the right.
Gradio 6 moved theme/css/js from gr.Blocks() to demo.launch(), and the
Chatbot now uses a messages-dict format instead of (user, assistant) tuples.
"""

import gradio as gr

from chain import answer_question
from ingest import ingest_pdfs
from utils import clear_database, format_sources, get_indexed_files, validate_pdf

# ── Helpers ────────────────────────────────────────────────────────────────────

def _indexed_footer_text() -> str:
    """Return a Markdown string listing currently indexed filenames."""
    try:
        files = get_indexed_files()
    except Exception:
        return "_Could not read index._"
    if not files:
        return "_No documents indexed yet._"
    return "📚 **Indexed:** " + "  ·  ".join(files)


# ── Event handlers ─────────────────────────────────────────────────────────────

def handle_index(uploaded_files):
    """Validate and ingest uploaded PDFs into ChromaDB."""
    if not uploaded_files:
        return "⚠️ No files selected. Upload at least one PDF.", _indexed_footer_text()

    # Gradio 6 with type='filepath' returns plain strings
    paths = []
    for f in uploaded_files:
        path = f if isinstance(f, str) else getattr(f, "path", getattr(f, "name", str(f)))
        ok, msg = validate_pdf(path)
        if not ok:
            return f"❌ {msg}", _indexed_footer_text()
        paths.append(path)

    try:
        total = ingest_pdfs(paths)
    except EnvironmentError as exc:
        return f"❌ Configuration error: {exc}", _indexed_footer_text()
    except Exception as exc:
        return f"❌ Indexing failed: {exc}", _indexed_footer_text()

    chunk_word = "chunk" if total == 1 else "chunks"
    file_word = "file" if len(paths) == 1 else "files"
    return (
        f"✅ Indexed {total} {chunk_word} from {len(paths)} {file_word}.",
        _indexed_footer_text(),
    )


def handle_send(message: str, history: list):
    """Call the QA chain and return updated history, sources markdown, cleared input."""
    if not message.strip():
        return history or [], "", ""

    history = list(history or [])

    try:
        result = answer_question(message)
    except EnvironmentError as exc:
        history += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ Configuration error: {exc}"},
        ]
        return history, "", ""
    except Exception as exc:
        history += [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"❌ Unexpected error: {exc}"},
        ]
        return history, "", ""

    confidence = result["confidence"]
    badge = {"high": "🟢 High", "medium": "🟡 Medium", "low": "🔴 Low"}.get(
        confidence, "⚪ Unknown"
    )
    answer_text = f"{result['answer']}\n\n_Confidence: {badge}_"

    history += [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer_text},
    ]
    return history, format_sources(result["sources"]), ""


def handle_clear_db():
    """Wipe the ChromaDB store and update the status and footer."""
    success, msg = clear_database()
    status = ("🗑️ " if success else "⚠️ ") + msg
    return status, _indexed_footer_text()


def handle_clear():
    """Reset the chatbot and sources panel."""
    return [], ""


# ── Theme / styling ────────────────────────────────────────────────────────────

_DARK_JS = "() => { document.documentElement.classList.add('dark'); }"

_CSS = """
#app-title  { text-align: center; padding: 0.4em 0 0.2em; }
#footer-md  { font-size: 0.78em; opacity: 0.65; margin-top: 0.6em; }
#sources-md { font-size: 0.88em; border-top: 1px solid #3a3a4a;
              padding-top: 0.6em; margin-top: 0.4em; min-height: 1.4em; }
"""

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="DocuMind — Chat with Your Documents") as demo:

    gr.Markdown(
        "# 🧠 DocuMind\n#### Chat with Your Documents",
        elem_id="app-title",
    )

    with gr.Row(equal_height=False):

        # ── Left column — ingestion ─────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### 📂 Ingest Documents")

            file_upload = gr.File(
                file_types=[".pdf"],
                file_count="multiple",
                label="Upload PDFs",
                type="filepath",
            )
            index_btn = gr.Button("📥 Index Documents", variant="primary", size="sm")
            clear_db_btn = gr.Button("🗑️ Clear Database", variant="stop", size="sm")
            status_msg = gr.Textbox(
                label="Status",
                interactive=False,
                lines=3,
                max_lines=3,
                placeholder="Upload PDFs and click 'Index Documents'…",
            )
            indexed_footer = gr.Markdown(
                value=_indexed_footer_text(),
                elem_id="footer-md",
            )

        # ── Right column — chat ─────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")

            chatbot = gr.Chatbot(
                value=[],
                height=430,
                show_label=False,
                layout="bubble",
                placeholder="Your answers will appear here…",
            )

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Ask anything about your documents…",
                    show_label=False,
                    lines=1,
                    max_lines=5,
                    scale=5,
                    container=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=80)

            sources_display = gr.Markdown(value="", elem_id="sources-md")
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")

    # ── Event wiring ────────────────────────────────────────────────────────

    index_btn.click(
        fn=handle_index,
        inputs=[file_upload],
        outputs=[status_msg, indexed_footer],
    )

    clear_db_btn.click(
        fn=handle_clear_db,
        outputs=[status_msg, indexed_footer],
    )

    send_btn.click(
        fn=handle_send,
        inputs=[user_input, chatbot],
        outputs=[chatbot, sources_display, user_input],
    )

    user_input.submit(
        fn=handle_send,
        inputs=[user_input, chatbot],
        outputs=[chatbot, sources_display, user_input],
    )

    clear_btn.click(
        fn=handle_clear,
        outputs=[chatbot, sources_display],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
        theme=gr.themes.Soft(),
        css=_CSS,
        js=_DARK_JS,
    )
