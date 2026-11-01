"""
Minimal RAG Web Application for DCCN Course Assistant
Reads and displays both text and diagrams from course PDFs using pypdf + Pillow + Gemini 2.5 Flash.
"""

import base64
import io
import os
import threading
import urllib.parse
import webbrowser
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import uvicorn

from pypdf import PdfReader
from PIL import Image
import chromadb
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer

load_dotenv()
if not os.getenv("GOOGLE_API_KEY") and not os.getenv("GEMINI_API_KEY"):
    load_dotenv("../ML/.env")

COURSE_DIR = "course"
CHROMA_PATH = "assistant_db"
COLLECTION_NAME = "my_three"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    max_output_tokens=1500,
)

app = FastAPI(title="Course Assistant")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    source: Optional[str] = None


def extract_page_diagrams(source_file: str, page_num: int) -> List[Tuple[int, bytes]]:
    """
    Extracts diagram and slide images from a PDF page using pypdf and Pillow.
    Filters out tiny icons/bullets and repeating syllabus banners.
    Returns a list of (image_index, png_bytes).
    """
    if "Syllabus" in source_file:
        return []

    pdf_path = os.path.join(COURSE_DIR, source_file)
    if not os.path.isfile(pdf_path):
        return []

    diagrams: List[Tuple[int, bytes]] = []
    try:
        reader = PdfReader(pdf_path)
        idx = int(page_num) - 1
        if idx < 0 or idx >= len(reader.pages):
            return []

        page = reader.pages[idx]
        for img_idx, img_obj in enumerate(page.images):
            try:
                pil_img = Image.open(io.BytesIO(img_obj.data))
                w, h = pil_img.size
                # Skip tiny icons/bullets (< 120x100)
                if w < 120 or h < 100:
                    continue
                buf = io.BytesIO()
                pil_img.convert("RGB").save(buf, format="PNG")
                diagrams.append((img_idx, buf.getvalue()))
            except Exception:
                continue
    except Exception:
        return []

    return diagrams


@app.get("/")
def serve_frontend():
    return FileResponse("index.html")


@app.get("/styles.css")
def serve_css():
    return FileResponse("styles.css", media_type="text/css")


@app.get("/script.js")
def serve_js():
    return FileResponse("script.js", media_type="application/javascript")


@app.get("/api/diagram/{source}/{page}/{img_idx}")
def serve_diagram(source: str, page: int, img_idx: int):
    """Serves a specific extracted diagram PNG from a PDF page."""
    safe_source = os.path.basename(urllib.parse.unquote(source))
    for idx, png_bytes in extract_page_diagrams(safe_source, page):
        if idx == img_idx:
            return Response(content=png_bytes, media_type="image/png")
    raise HTTPException(status_code=404, detail="Diagram not found")


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    q_embedding = embed_model.encode([query]).tolist()
    where_filter = {"Source": req.source} if req.source else None

    results = collection.query(
        query_embeddings=q_embedding,
        n_results=3,
        where=where_filter,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    if not docs:
        return {
            "answer": "I could not find that information in the provided source.",
            "sources": [],
            "diagrams": [],
        }

    context = "\n\n".join(docs)
    prompt_text = f"""You are a question-answering course assistant for Data Communication & Computer Networks (DCCN).

Answer the question using ONLY the provided text context and any attached diagrams from the course slides.
- If diagrams are attached, read and explain the visual steps, labels, or architecture shown in them.
- If the answer is not present in the context or diagrams, say:
  "I could not find that information in the provided source."

Context:
{context}

Question:
{query}

Answer:"""

    message_content = [{"type": "text", "text": prompt_text}]
    sources = []
    diagrams = []
    seen_pages = set()

    for m in metas:
        src = m.get("Source", "Unknown")
        pg = m.get("page_number", 1)
        sources.append({"source": src, "page": pg})

        page_key = (src, pg)
        if page_key not in seen_pages:
            seen_pages.add(page_key)
            page_diagrams = extract_page_diagrams(src, pg)
            encoded_src = urllib.parse.quote(src)
            for img_idx, png_bytes in page_diagrams:
                b64_img = base64.b64encode(png_bytes).decode("utf-8")
                message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    }
                )
                diagrams.append(
                    {
                        "source": src,
                        "page": pg,
                        "url": f"/api/diagram/{encoded_src}/{pg}/{img_idx}",
                    }
                )

    response = llm.invoke([HumanMessage(content=message_content)])

    return {
        "answer": response.content,
        "sources": sources,
        "diagrams": diagrams,
    }


if __name__ == "__main__":
    url = "http://127.0.0.1:8000"
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)
