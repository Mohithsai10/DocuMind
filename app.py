"""
app.py — Gradio 6.x web interface for DocuMind.
Daylight-inspired warm editorial theme, with readable dark text. Logic untouched.
"""

import gradio as gr

from chain import answer_question
from ingest import ingest_pdfs
from utils import clear_database, format_sources, get_indexed_files, validate_pdf

EXAMPLE_QUERIES = [
    "What are the 3 key findings in this document?",
    "Summarize this document in five bullet points.",
    "What certifications or credentials are mentioned?",
    "List every date and what happened on it.",
]


def _indexed_footer_text() -> str:
    try:
        files = get_indexed_files()
    except Exception:
        return "_Index unavailable._"
    if not files:
        return "No documents indexed yet. Upload a PDF to begin."
    label = "document" if len(files) == 1 else "documents"
    return f"**{len(files)} {label} indexed** &nbsp;·&nbsp; " + "  ·  ".join(files)


def handle_index(uploaded_files):
    if not uploaded_files:
        return "⚠️ No files selected. Upload at least one PDF.", _indexed_footer_text()

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
        f"✅ Indexed {total} {chunk_word} from {len(paths)} {file_word}. Ask a question →",
        _indexed_footer_text(),
    )


def handle_send(message: str, history: list):
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
    success, msg = clear_database()
    status = ("🗑️ " if success else "⚠️ ") + msg
    return status, _indexed_footer_text()


def handle_clear():
    return [], ""


def use_example(example: str):
    return example


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.stone,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(
    body_background_fill="#faf3e7",
    body_background_fill_dark="#faf3e7",
    background_fill_primary="#fffdf8",
    background_fill_primary_dark="#fffdf8",
    background_fill_secondary="#f4ead6",
    background_fill_secondary_dark="#f4ead6",
    border_color_primary="#e4d7bf",
    border_color_primary_dark="#e4d7bf",
    block_background_fill="#fffdf8",
    block_border_color="#e4d7bf",
    block_label_text_color="#7a6f5c",
    block_title_text_color="#1c1a17",
    body_text_color="#26231e",
    body_text_color_subdued="#7a6f5c",
    button_primary_background_fill="#e8792b",
    button_primary_background_fill_hover="#d76a1f",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#1c1a17",
    button_secondary_text_color="#faf3e7",
    input_background_fill="#fffdf8",
    input_border_color="#e4d7bf",
    input_border_color_focus="#e8792b",
)

_CSS = """
:root { --dl-accent:#e8792b; --dl-cream:#faf3e7; --dl-line:#e4d7bf;
        --dl-ink:#1c1a17; --dl-muted:#7a6f5c; }
.gradio-container { max-width: 1200px !important; margin: 0 auto !important;
  background: var(--dl-cream) !important; }
#dl-hero { position:relative; text-align:center; padding: 40px 0 20px;
  border-bottom: 1px solid var(--dl-line); margin-bottom: 22px; }
#dl-hero .dl-eyebrow {
  font-family:'JetBrains Mono', monospace; font-size:11px; letter-spacing:4px;
  text-transform:uppercase; color:var(--dl-muted); margin-bottom:16px;
}
#dl-hero .dl-name {
  font-family: Georgia, 'Times New Roman', serif; font-weight:600;
  font-size:56px; line-height:0.95; letter-spacing:-1.5px; color:var(--dl-ink);
  margin:0;
}
#dl-hero .dl-name em { font-style:italic; color:var(--dl-accent); }
#dl-hero .dl-tag { color:var(--dl-muted); font-size:16px; margin-top:14px;
  max-width:520px; margin-left:auto; margin-right:auto; line-height:1.5; }
.dl-panel {
  background:#fffdf8; border:1px solid var(--dl-line); border-radius:14px;
  padding:20px 20px 22px !important;
}
.dl-col-label {
  font-family:'JetBrains Mono', monospace; font-size:11px; letter-spacing:2.5px;
  text-transform:uppercase; color:var(--dl-muted); margin:2px 0 14px;
}
#dl-footer { font-size:12.5px; color:var(--dl-muted); margin-top:14px;
  padding-top:14px; border-top:1px solid var(--dl-line); line-height:1.6; }
#dl-sources { font-size:13px; background:#f9f0dd; border:1px solid var(--dl-line);
  border-radius:12px; padding:12px 14px; margin-top:10px; min-height:1.2em;
  color:#3a352c; }
#dl-sources:empty::before { content:'Sources will appear here after you ask a question.';
  color:var(--dl-muted); font-style:italic; }
#dl-examples { gap:8px !important; margin:4px 0 2px; }
#dl-examples button {
  background:#fffdf8 !important; border:1px solid var(--dl-line) !important;
  color:#4a4234 !important; font-size:12.5px !important; font-weight:500 !important;
  border-radius:999px !important; padding:7px 13px !important; text-align:left !important;
  transition:all .15s ease;
}
#dl-examples button:hover {
  border-color:var(--dl-accent) !important; color:var(--dl-accent) !important;
  transform:translateY(-1px);
}
.dl-hint { font-family:'JetBrains Mono',monospace; font-size:11px; letter-spacing:2px;
  text-transform:uppercase; color:var(--dl-muted); margin:10px 0 2px; }
#dl-send { border-radius:10px !important; font-weight:600 !important; }
#dl-chat, #dl-chat * { color:#1c1a17 !important; }
#dl-chat .message, #dl-chat [class*="bubble"], #dl-chat [class*="message"] {
  background:#fffdf8 !important; color:#1c1a17 !important;
  border:1px solid #e4d7bf !important;
}
#dl-chat .placeholder, #dl-chat [class*="placeholder"] { color:#7a6f5c !important; }
#dl-chat em, #dl-chat i { color:#7a6f5c !important; }
.dl-panel [class*="upload"], .dl-panel [class*="Upload"] {
  background:#fffdf8 !important; color:#1c1a17 !important;
}
.dl-panel [class*="upload"] * { color:#1c1a17 !important; }
.dl-panel textarea, .dl-panel input, #dl-userbox textarea {
  color:#1c1a17 !important; background:#fffdf8 !important;
}
.dl-panel textarea::placeholder, #dl-userbox textarea::placeholder { color:#7a6f5c !important; }
#dl-sources, #dl-sources * { color:#3a352c !important; }
#dl-chat p, #dl-chat li, #dl-chat span, #dl-chat div { color:#1c1a17 !important; }
footer { display:none !important; }
"""

with gr.Blocks(title="DocuMind — Chat with Your Documents") as demo:

    gr.HTML(
        """
        <div id="dl-hero">
          <div class="dl-eyebrow">Retrieval-Augmented Document Intelligence</div>
          <h1 class="dl-name">Docu<em>Mind</em></h1>
          <div class="dl-tag">Upload your PDFs and get answers with cited page numbers — nothing invented.</div>
        </div>
        """
    )

    with gr.Row(equal_height=False):

        with gr.Column(scale=2, min_width=300):
            with gr.Group(elem_classes="dl-panel"):
                gr.HTML('<div class="dl-col-label">01 — Ingest documents</div>')

                file_upload = gr.File(
                    file_types=[".pdf"],
                    file_count="multiple",
                    label="Upload PDFs",
                    type="filepath",
                )
                index_btn = gr.Button("Index documents", variant="primary", size="sm")
                clear_db_btn = gr.Button("Clear database", variant="stop", size="sm")
                status_msg = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3,
                    max_lines=3,
                    placeholder="Upload PDFs, then click “Index documents.”",
                )
                indexed_footer = gr.Markdown(
                    value=_indexed_footer_text(),
                    elem_id="dl-footer",
                )

        with gr.Column(scale=3):
            with gr.Group(elem_classes="dl-panel"):
                gr.HTML('<div class="dl-col-label">02 — Ask questions</div>')

                chatbot = gr.Chatbot(
                    value=[],
                    height=390,
                    show_label=False,
                    layout="bubble",
                    elem_id="dl-chat",
                    placeholder="Answers with cited sources will appear here.",
                )

                gr.HTML('<div class="dl-hint">Try one</div>')
                with gr.Row(elem_id="dl-examples"):
                    example_btns = [
                        gr.Button(q, size="sm", variant="secondary") for q in EXAMPLE_QUERIES
                    ]

                with gr.Row():
                    user_input = gr.Textbox(
                        placeholder="Ask anything about your documents…",
                        show_label=False,
                        lines=1,
                        max_lines=5,
                        scale=5,
                        container=False,
                        elem_id="dl-userbox",
                    )
                    send_btn = gr.Button(
                        "Send ➤", variant="primary", scale=1, min_width=90, elem_id="dl-send"
                    )

                sources_display = gr.Markdown(value="", elem_id="dl-sources")
                clear_btn = gr.Button("Clear chat", variant="secondary", size="sm")

    index_btn.click(fn=handle_index, inputs=[file_upload], outputs=[status_msg, indexed_footer])
    clear_db_btn.click(fn=handle_clear_db, outputs=[status_msg, indexed_footer])
    send_btn.click(fn=handle_send, inputs=[user_input, chatbot],
                   outputs=[chatbot, sources_display, user_input])
    user_input.submit(fn=handle_send, inputs=[user_input, chatbot],
                      outputs=[chatbot, sources_display, user_input])
    clear_btn.click(fn=handle_clear, outputs=[chatbot, sources_display])
    for btn in example_btns:
        btn.click(fn=use_example, inputs=[btn], outputs=[user_input])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True,
                share=False, theme=theme, css=_CSS)
