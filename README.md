# DCCN Course Assistant

A study assistant for **Data Communication and Computer Networks (DCCN)** that answers student questions **only from the professor-provided course PDFs**, and shows the exact file and page each answer came from.

<img width="1917" height="892" alt="Screenshot 2026-11-01 183423" src="https://github.com/user-attachments/assets/a94db6cb-c2de-4baa-9a28-ea9a7058df29" />
<img width="1917" height="941" alt="Screenshot 2026-11-01 183507" src="https://github.com/user-attachments/assets/ce083815-84d3-4262-93d8-c90af02742f9" />


---

## The problem

Before exams, students jump between several ChatGPT or Gemini chats to study one subject. Those general chatbots often drift off-topic, go beyond the syllabus, or give answers that don't match what the professor taught. Students also end up scrolling through long module PDFs to find one definition.

## The solution

One focused assistant for the course:

- Answers are **grounded in the professor-provided module PDFs**, not the open internet.
- Every answer shows **source chips** (e.g. `Mod3_DCCN.pdf · p.9`) so students can open the exact page and verify.
- A **module filter** lets students restrict a question to a single module or search all modules.
- Lets students ask follow-up questions in one place instead of juggling multiple chat windows.

## Screenshots

| Explaining a topic | Answer with page citations |
|---|---|
| ![TCP segment](screenshots/tcp-segment.png) | ![UDP with sources](screenshots/udp-answer.png) |

## Features

- Question answering over course PDFs (Modules [1-X])
- Page-level source citations on every answer
- Module selector (All Modules or a specific module)
- Clear-chat button for a fresh study session
- [Refuses to answer when the topic isn't in the course material] *(keep only if true and tested)*
- [Streaming responses] *(keep only if true)*

## How it works

```
Course PDFs
   |  text extraction (page numbers kept as metadata)
   v
Chunking  -->  Sentence Transformer embeddings  -->  ChromaDB
                                                        ^
Student question  -->  embed  -->  top-k retrieval -----+
                                        |
                          retrieved chunks + question
                                        v
                                     Gemini
                                        |
                      answer + source chips (file, page)
```

1. **Ingestion:** PDFs are parsed page by page, split into chunks, and each chunk is stored with its source file and page number.
2. **Retrieval:** The question is embedded and the top-k most similar chunks are fetched from ChromaDB (optionally filtered by module).
3. **Generation:** Gemini receives only the retrieved chunks and is instructed to answer from them.
4. **Citations:** The files and pages of the retrieved chunks are shown under the answer.

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python |
| Embeddings | Sentence Transformers (`[model name, e.g. all-MiniLM-L6-v2]`) |
| Vector database | ChromaDB |
| LLM | Google Gemini (`[model name]`) |
| Backend | `[FastAPI / Flask]` |
| Frontend | `[HTML, CSS, JavaScript]` |
| Deployment | `[Render / Hugging Face Spaces / other]` |

## Project structure

```
.
├── data/               # course PDFs (or instructions to add yours)
├── ingest.py           # extract, chunk, embed, store in ChromaDB
├── app.py              # backend server
├── static/ or templates/   # frontend
├── screenshots/
├── requirements.txt
├── .env.example
└── README.md
```
*(Update this to match your real folders.)*

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/MohammedBaqar0223/YOUR_REPO.git
cd YOUR_REPO
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add your API key

Create a `.env` file (see `.env.example`):

```
GEMINI_API_KEY=your_key_here
```

### 3. Add course PDFs and build the index

Place your PDFs in `data/`, then run:

```bash
python ingest.py
```

### 4. Run the app

```bash
python app.py
```

Open `http://localhost:[PORT]` in your browser.

## Example questions

- Explain the TCP segment
- Explain UDP
- What is the difference between TCP and UDP?
- [Add 2-3 more from your modules]

## Results *(optional, add only if you measured)*

| Metric | Value |
|---|---|
| Test questions | [X] |
| Recall@5 (correct page in top 5 chunks) | [X]% |
| Used by | [X] classmates |

## Limitations

- Answers are only as good as the uploaded PDFs; diagrams and tables inside PDFs may not be extracted well.
- [Add anything else you've noticed, e.g. scanned pages, formulas.]

## Future improvements

- Hybrid search (keyword + semantic) and reranking
- Evaluation set to track retrieval quality
- Support for more subjects and previous-year question papers
- Feedback buttons (thumbs up/down) on answers

## Author

**Mohammed Baqar**, B.Tech Information Technology, CMR University
[GitHub](https://github.com/MohammedBaqar0223) | [LinkedIn](YOUR_LINKEDIN) | mohammedbaqar9658@gmail.com
