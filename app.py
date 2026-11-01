"""
DCCN AI Course Assistant - Interactive RAG Web Application
Built with Gradio 6, ChromaDB, SentenceTransformers, and Google Gemini.
"""

import os
import time
from typing import Generator, List, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# Also check parent directory if needed
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    load_dotenv("../ML/.env")

import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
import gradio as gr

# --- Initialize Core Services ---
CHROMA_PATH = "assistant_db"
COLLECTION_NAME = "my_three"

print("Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print(f"Connecting to ChromaDB at '{CHROMA_PATH}'...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    chunk_count = collection.count()
    print(f"ChromaDB ready with {chunk_count} document chunks.")
except Exception as e:
    print(f"Warning: Could not open collection '{COLLECTION_NAME}': {e}")
    collection = None
    chunk_count = 0


def get_llm(temperature: float = 0.2):
    """Factory function for Gemini LLM instance."""
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
        max_output_tokens=1500,
    )


# Mapping friendly module names to source filenames in Chroma
MODULE_MAP = {
    "🌐 All Course Modules": None,
    "📋 Syllabus (3. DCCN Syllabus.pdf)": "3. DCCN Syllabus.pdf",
    "🔌 Module 1: Physical Layer (Mod1_DCCN.pdf)": "Mod1_DCCN.pdf",
    "📦 Module 2: Data Link Layer (Mod2_DCCN.pdf)": "Mod2_DCCN.pdf",
    "📡 Module 3: MAC & LANs (Mod3_DCCN.pdf)": "Mod3_DCCN.pdf",
}

# --- Core RAG Function ---
def search_and_generate(
    query: str,
    history: List,
    module_filter: str,
    top_k: int,
    temperature: float,
    show_sources: bool,
) -> Generator[List[dict], None, None]:
    """Retrieves relevant chunks and streams formatted response with citations."""
    if not query or not query.strip():
        yield history
        return

    # Add user message to history
    user_msg = query.strip()
    history = history or []
    history = list(history)
    history.append({"role": "user", "content": user_msg})
    # Placeholder for assistant response
    history.append({"role": "assistant", "content": "Thinking and retrieving course notes..."})
    yield history

    if collection is None:
        history[-1]["content"] = "⚠️ **Error:** ChromaDB collection could not be loaded."
        yield history
        return

    # 1. Encode query
    query_emb = embed_model.encode([user_msg]).tolist()

    # 2. Query Chroma with optional filter
    where_filter = None
    source_filename = MODULE_MAP.get(module_filter)
    if source_filename:
        where_filter = {"Source": source_filename}

    try:
        results = collection.query(
            query_embeddings=query_emb,
            n_results=int(top_k),
            where=where_filter if where_filter else None,
        )
    except Exception as err:
        history[-1]["content"] = f"⚠️ **Retrieval Error:** {str(err)}"
        yield history
        return

    retrieved_docs = results.get("documents", [[]])[0]
    retrieved_metas = results.get("metadatas", [[]])[0]

    if not retrieved_docs:
        history[-1]["content"] = (
            "I could not find relevant information in the selected course materials. "
            "Please try rephrasing your question or selecting '🌐 All Course Modules'."
        )
        yield history
        return

    # 3. Format Context for LLM
    context_blocks = []
    citations_data = []
    for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas), 1):
        source_name = meta.get("Source", "Course Document")
        page_no = meta.get("page_number", "?")
        context_blocks.append(f"--- [Source #{idx}: {source_name} (Page {page_no})] ---\n{doc}")
        citations_data.append({
            "index": idx,
            "source": source_name,
            "page": page_no,
            "snippet": doc.strip()
        })

    combined_context = "\n\n".join(context_blocks)

    # 4. Construct Prompt
    system_prompt = f"""You are an expert, encouraging professor and teaching assistant for the course 'Data Communication & Computer Networks' (DCCN).
Answer the student's question clearly, thoroughly, and academically using ONLY the provided course excerpts.

Guidelines:
1. Explain concepts step-by-step with intuitive analogies where appropriate.
2. Structure your answer using markdown with clear headings, bullet points, and bold text for key terms.
3. If equations, protocol steps, or frames are discussed, use clean bullet points or code blocks.
4. If the exact answer is not contained in the provided excerpts, clearly state:
   "I could not find that specific detail in the provided course slides/syllabus."
5. Do NOT make up facts outside the course context.

Course Context Excerpts:
{combined_context}

Student Question:
{user_msg}

Answer:"""

    # 5. Stream response from Gemini
    try:
        llm = get_llm(temperature=float(temperature))
        collected_response = ""
        for chunk in llm.stream(system_prompt):
            if chunk and chunk.content:
                collected_response += chunk.content
                # Format current output
                history[-1]["content"] = collected_response
                yield history

        # Append beautiful citations section if enabled
        if show_sources and citations_data:
            citations_html = "\n\n---\n### 📖 Verified Course References & Slide Citations\n"
            citations_html += "<div class='sources-container'>\n"
            for cite in citations_data:
                snippet_preview = cite["snippet"]
                if len(snippet_preview) > 280:
                    snippet_preview = snippet_preview[:280] + "..."
                snippet_preview = snippet_preview.replace("\n", " ")
                
                citations_html += f"""
<div class='source-card'>
    <div class='source-header'>
        <span class='source-badge'>📄 {cite['source']}</span>
        <span class='page-badge'>📌 Slide / Page {cite['page']}</span>
    </div>
    <div class='source-snippet'>"{snippet_preview}"</div>
</div>
"""
            citations_html += "</div>\n"
            collected_response += citations_html
            history[-1]["content"] = collected_response
            yield history

    except Exception as err:
        history[-1]["content"] = (
            f"⚠️ **Error generating response:** {str(err)}\n\n"
            "Please ensure your Gemini/Google API key is configured properly."
        )
        yield history


def clear_chat():
    """Reset the chat history."""
    return []


def populate_question(prompt_text: str):
    """Callback to place an example question into the textbox."""
    return prompt_text


# --- Custom Styling & Theme ---
custom_css = """
/* Import Modern Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Inter:wght@400;500;600&family=Fira+Code:wght@400;500&display=swap');

:root {
    --primary-gradient: linear-gradient(135deg, #4f46e5 0%, #3b82f6 50%, #06b6d4 100%);
    --card-bg-glass: rgba(255, 255, 255, 0.7);
    --border-glass: rgba(226, 232, 240, 0.8);
    --shadow-soft: 0 10px 25px -5px rgba(59, 130, 246, 0.08), 0 8px 10px -6px rgba(59, 130, 246, 0.04);
}

body, .gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

h1, h2, h3, h4, .app-header {
    font-family: 'Outfit', sans-serif !important;
}

/* App Header Card */
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
    border-radius: 16px;
    padding: 24px 28px;
    color: white;
    margin-bottom: 16px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    overflow: hidden;
}

.header-box::after {
    content: '';
    position: absolute;
    top: -50px;
    right: -50px;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.3) 0%, rgba(0, 0, 0, 0) 70%);
    border-radius: 50%;
    pointer-events: none;
}

.header-title {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: -0.5px;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-subtitle {
    font-size: 14px;
    color: #94a3b8;
    margin-top: 6px;
    max-width: 650px;
    line-height: 1.5;
}

.status-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #f1f5f9;
}

.badge.pulse-active {
    background: rgba(16, 185, 129, 0.15);
    border-color: rgba(16, 185, 129, 0.3);
    color: #34d399;
}

.pulse-dot {
    width: 8px;
    height: 8px;
    background-color: #10b981;
    border-radius: 50%;
    box-shadow: 0 0 8px #10b981;
    animation: pulseAnim 2s infinite;
}

@keyframes pulseAnim {
    0% { transform: scale(0.95); opacity: 0.8; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(0.95); opacity: 0.8; }
}

/* Sidebar Styling */
.sidebar-panel {
    background: var(--card-bg-glass);
    border-radius: 14px;
    padding: 16px;
    border: 1px solid var(--border-glass);
    box-shadow: var(--shadow-soft);
}

/* Quick prompt buttons */
.quick-prompt-btn {
    text-align: left !important;
    justify-content: flex-start !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    transition: all 0.2s ease !important;
    color: #334155 !important;
}

.quick-prompt-btn:hover {
    background: #eef2ff !important;
    border-color: #c7d2fe !important;
    color: #4338ca !important;
    transform: translateX(3px);
}

/* Chatbot container improvements */
#course-chatbot {
    border-radius: 14px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.03) !important;
    min-height: 540px !important;
}

/* Custom Citation Cards */
.sources-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 12px;
}

.source-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #4f46e5;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    transition: background 0.2s ease;
}

.source-card:hover {
    background: #f1f5f9;
}

.source-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.source-badge {
    font-weight: 600;
    color: #312e81;
    background: #e0e7ff;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11.5px;
}

.page-badge {
    font-weight: 600;
    color: #0369a1;
    background: #e0f2fe;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 11.5px;
}

.source-snippet {
    color: #475569;
    font-style: italic;
    font-size: 12px;
    line-height: 1.4;
}

/* Primary Action Button */
.action-btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}

.action-btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important;
}

/* Secondary Button */
.action-btn-secondary {
    border-radius: 10px !important;
    font-weight: 500 !important;
}
"""

theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

# --- Build Gradio Interface ---
with gr.Blocks(title="DCCN AI Course Companion") as demo:
    # 1. Header Section
    gr.HTML(f"""
    <div class="header-box">
        <h1 class="header-title">🎓 DCCN AI Course Companion</h1>
        <p class="header-subtitle">
            Instant, cited answers grounded in your official lecture slides and syllabus for 
            <b>Data Communication & Computer Networks</b>.
        </p>
        <div class="status-badges">
            <span class="badge pulse-active"><span class="pulse-dot"></span> Vector DB Active ({chunk_count} Chunks)</span>
            <span class="badge">🤖 Model: Gemini 2.5 Flash</span>
            <span class="badge">📚 4 Course PDFs Indexed</span>
            <span class="badge">⚡ RAG Grounded</span>
        </div>
    </div>
    """)

    # 2. Main Two-Column Layout
    with gr.Row():
        # LEFT COLUMN: Sidebar with Scope, Settings, and Quick Questions
        with gr.Column(scale=1, min_width=320, elem_classes=["sidebar-panel"]):
            gr.Markdown("### 🎯 **Search Scope**")
            module_selector = gr.Dropdown(
                choices=list(MODULE_MAP.keys()),
                value="🌐 All Course Modules",
                label="Filter by Course Module",
                info="Target your query to a specific module or search everything.",
            )

            with gr.Accordion("⚙️ **RAG Retrieval Settings**", open=False):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=6,
                    value=3,
                    step=1,
                    label="Retrieved Passages (Top-K)",
                    info="Higher = more context, slightly slower.",
                )
                temp_slider = gr.Slider(
                    minimum=0.0,
                    maximum=0.7,
                    value=0.2,
                    step=0.05,
                    label="Temperature (Creativity)",
                    info="Lower values produce more factual, grounded answers.",
                )
                show_sources_check = gr.Checkbox(
                    value=True,
                    label="Show Slide Excerpts & Citations",
                    info="Displays exact source filename and page numbers.",
                )

            gr.Markdown("### 💡 **Quick Study Prompts**")
            gr.Markdown("<p style='font-size:12px; color:#64748b;'>Click any prompt to instantly ask the assistant:</p>")

            q1_btn = gr.Button("📋 What is the syllabus for Module 3?", elem_classes=["quick-prompt-btn"])
            q2_btn = gr.Button("🔄 Stop-and-Wait vs Sliding Window protocols", elem_classes=["quick-prompt-btn"])
            q3_btn = gr.Button("🛡️ What are the responsibilities of Data Link Layer?", elem_classes=["quick-prompt-btn"])
            q4_btn = gr.Button("📶 Explain CSMA/CD and CSMA/CA differences", elem_classes=["quick-prompt-btn"])
            q5_btn = gr.Button("📡 Compare guided vs unguided transmission media", elem_classes=["quick-prompt-btn"])

            with gr.Accordion("📖 **Course Materials Breakdown**", open=False):
                gr.Markdown("""
                - **`3. DCCN Syllabus.pdf`**: Course overview, topics, reference books.
                - **`Mod1_DCCN.pdf`**: Physical layer, topologies, transmission media.
                - **`Mod2_DCCN.pdf`**: Data link layer, framing, flow/error control, HDLC.
                - **`Mod3_DCCN.pdf`**: Multiple access, CSMA, IEEE 802 LANs, Ethernet.
                """)

        # RIGHT COLUMN: Chatbot and Conversation Area
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                elem_id="course-chatbot",
                placeholder="""## 🎓 Welcome to the DCCN AI Course Companion!
Ask any question about your course materials, for example:
- *"What topics are included in the syllabus for Module 2?"*
- *"Explain CRC (Cyclic Redundancy Check) and how it detects errors."*
- *"What is the difference between pure ALOHA and slotted ALOHA?"*

💡 *Tip: Check the sidebar to filter searches by specific modules or adjust retrieval depth.*
""",
                height=560,
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about DCCN slides or syllabus...",
                    show_label=False,
                    scale=5,
                    autofocus=True,
                    lines=1,
                    max_lines=4,
                )
                submit_btn = gr.Button("Ask Question 🚀", scale=1, elem_classes=["action-btn-primary"])

            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Conversation", size="sm", elem_classes=["action-btn-secondary"])
                retry_btn = gr.Button("🔄 Regenerate Last Answer", size="sm", elem_classes=["action-btn-secondary"])

    # 3. Interactivity & Event Handlers
    # Submit via Enter key or Ask button
    submit_event = msg_input.submit(
        fn=search_and_generate,
        inputs=[msg_input, chatbot, module_selector, top_k_slider, temp_slider, show_sources_check],
        outputs=[chatbot],
    ).then(
        fn=lambda: "",
        outputs=[msg_input],
    )

    submit_btn.click(
        fn=search_and_generate,
        inputs=[msg_input, chatbot, module_selector, top_k_slider, temp_slider, show_sources_check],
        outputs=[chatbot],
    ).then(
        fn=lambda: "",
        outputs=[msg_input],
    )

    # Clear chat
    clear_btn.click(fn=clear_chat, outputs=[chatbot])

    # Retry: re-run the last user prompt
    def retry_last(history, module_filter, top_k, temp, show_sources):
        if not history:
            yield history
            return
        # Find last user message
        last_user_msg = ""
        for item in reversed(history):
            if isinstance(item, dict) and item.get("role") == "user":
                last_user_msg = item.get("content", "")
                break
        if not last_user_msg:
            yield history
            return
        
        # Pop assistant message if present
        trimmed_hist = [h for h in history if h != history[-1]] if history[-1].get("role") == "assistant" else history
        trimmed_hist = [h for h in trimmed_hist if h != trimmed_hist[-1]]  # remove user message too, since search_and_generate will re-add it
        yield from search_and_generate(last_user_msg, trimmed_hist, module_filter, top_k, temp, show_sources)

    retry_btn.click(
        fn=retry_last,
        inputs=[chatbot, module_selector, top_k_slider, temp_slider, show_sources_check],
        outputs=[chatbot],
    )

    # Quick prompt buttons click handlers
    for btn, prompt_text in [
        (q1_btn, "What is the syllabus for Module 3?"),
        (q2_btn, "Compare Stop-and-Wait vs Sliding Window protocols"),
        (q3_btn, "What are the responsibilities of the Data Link Layer?"),
        (q4_btn, "Explain the differences between CSMA/CD and CSMA/CA protocols"),
        (q5_btn, "Compare guided vs unguided transmission media"),
    ]:
        btn.click(
            fn=lambda p=prompt_text: p,
            outputs=[msg_input],
        )

# Launch application
if __name__ == "__main__":
    demo.launch(
        theme=theme,
        css=custom_css,
        inbrowser=True,
        share=False,
    )
