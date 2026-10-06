# Course_Assistant
# 📚 RAG-Based Course Assistant

A simple **Retrieval-Augmented Generation (RAG)** application that allows users to ask questions about course materials stored as PDF files.

The application extracts text from PDFs, splits the content into smaller chunks, converts the chunks into vector embeddings, stores them in a **ChromaDB vector database**, retrieves the most relevant information for a user's question, and uses **Google Gemini** to generate an answer based only on the retrieved course content.

A **Gradio chat interface** is provided so users can interact with the course assistant easily.

---

## 🚀 Features

- 📄 Extracts text from multiple PDF course materials
- ✂️ Splits large documents into smaller overlapping chunks
- 🧠 Generates semantic embeddings using `all-MiniLM-L6-v2`
- 🗄️ Stores embeddings in a persistent ChromaDB vector database
- 🔍 Performs semantic search to retrieve relevant course content
- 🤖 Uses Google Gemini 2.5 Flash to generate answers
- 💬 Provides an interactive Gradio chatbot interface
- 📌 Maintains source file and page number metadata
- 🚫 Restricts the LLM to answering from the retrieved course material

---

## 🧠 How It Works

The project follows a standard Retrieval-Augmented Generation pipeline:

```text
PDF Course Materials
        ↓
Text Extraction
        ↓
Text Chunking
        ↓
Generate Embeddings
        ↓
Store in ChromaDB
        ↓
User Question
        ↓
Question Embedding
        ↓
Similarity Search
        ↓
Retrieve Relevant Chunks
        ↓
Gemini LLM
        ↓
Generated Answer
```

---

## 🛠️ Technologies Used

- **Python**
- **PyPDF**
- **LangChain Text Splitters**
- **Sentence Transformers**
- **ChromaDB**
- **Google Gemini API**
- **LangChain Google GenAI**
- **Gradio**
- **python-dotenv**

---

## 📂 Project Structure

```text
project/
│
├── first.ipynb
├── course/
│   ├── module1.pdf
│   ├── module2.pdf
│   └── ...
│
├── assistant_db/
│
├── .env
│
└── README.md
```

### `course/`

Place all course PDF files inside this folder. The application automatically reads every `.pdf` file present in the directory.

### `assistant_db/`

Stores the persistent ChromaDB vector database generated from the course material.

### `.env`

Stores the Google Gemini API key securely.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Install the required libraries

```bash
pip install pypdf
pip install langchain-text-splitters
pip install sentence-transformers
pip install chromadb
pip install langchain-google-genai
pip install python-dotenv
pip install gradio
```

Or install everything together:

```bash
pip install pypdf langchain-text-splitters sentence-transformers chromadb langchain-google-genai python-dotenv gradio
```

---

## 🔑 Configure Gemini API

Create a `.env` file in the project directory.

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Do not upload your `.env` file to GitHub.

Add it to `.gitignore`:

```text
.env
```

---

## 📄 Add Course Material

Create a folder named:

```text
course
```

Place your course PDFs inside it.

Example:

```text
course/
├── Computer_Networks_Module1.pdf
├── Computer_Networks_Module2.pdf
└── Computer_Networks_Module3.pdf
```

The notebook automatically loops through all PDF files in this folder and extracts their text.

---

## 🔍 Document Processing

The PDF files are first read using **PyPDF**.

Each extracted page is stored along with:

- Source PDF filename
- Page number
- Extracted text

The text is then divided using LangChain's `RecursiveCharacterTextSplitter`.

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
```

The overlap helps preserve context between neighboring chunks.

---

## 🧠 Embedding Generation

The project uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

Each text chunk is converted into a numerical vector representation.

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = model.encode(text)
```

These embeddings allow the system to search based on **semantic similarity** rather than exact keyword matching.

---

## 🗄️ Vector Database

The generated embeddings are stored using **ChromaDB**.

```python
client = chromadb.PersistentClient(
    path="assistant_db"
)
```

Along with each embedding, the project stores:

- Original text
- Source PDF
- Page number

This allows relevant course information to be retrieved later.

---

## 🔎 Retrieval

When the user asks a question, the question is converted into an embedding using the same Sentence Transformer model.

```python
q_embedding = model.encode([query])
```

ChromaDB then retrieves the **three most relevant chunks**:

```python
results = collection.query(
    query_embeddings=q_embedding.tolist(),
    n_results=3
)
```

These chunks become the context supplied to the LLM.

---

## 🤖 Answer Generation

The retrieved context and user question are passed to:

```text
Gemini 2.5 Flash
```

The assistant is instructed to answer only using the supplied course material.

If the information cannot be found, the assistant responds:

```text
I could not find that information in the provided source.
```

This helps reduce hallucinations and keeps the answers grounded in the uploaded material.

---

## 💬 Gradio Interface

The project includes a simple chat interface using Gradio.

```python
gr.ChatInterface(
    fn=answer_question,
    title="Course Assistant",
).launch(inbrowser=True)
```

After running the notebook, a browser window opens where questions can be asked directly.

Example questions:

```text
What is the syllabus in module 3?
```

```text
Explain TCP congestion control.
```

```text
What topics are covered in module 2?
```

The RAG system searches the course material and generates an answer based on the most relevant retrieved content.

---

## 🏗️ RAG Architecture

The application contains two major stages.

### Indexing Stage

```text
PDF
 ↓
Extract Text
 ↓
Chunk Text
 ↓
Sentence Transformer
 ↓
Embeddings
 ↓
ChromaDB
```

This stage prepares the course content for semantic search.

### Query Stage

```text
User Question
 ↓
Question Embedding
 ↓
ChromaDB Similarity Search
 ↓
Top Relevant Chunks
 ↓
Gemini 2.5 Flash
 ↓
Final Answer
```

---

## 🎯 Purpose of the Project

This project demonstrates the fundamental concepts behind modern **RAG applications**, including:

- Document ingestion
- Text preprocessing
- Chunking
- Embeddings
- Vector databases
- Semantic search
- Context retrieval
- Prompt construction
- Large Language Model integration
- Building a chatbot interface

It can be extended into applications such as:

- AI study assistants
- College notes chatbots
- Document question-answering systems
- Research paper assistants
- Company knowledge-base assistants
- PDF chat applications

---

## 🔮 Future Improvements

Possible improvements include:

- Displaying source PDF names in the final answer
- Displaying page numbers used for each answer
- Supporting PDF uploads directly through Gradio
- Adding conversation memory
- Improving retrieval using reranking
- Using hybrid keyword + vector search
- Supporting multiple subjects or courses
- Adding authentication
- Deploying the application online
- Implementing evaluation metrics for RAG quality

---

## 👨‍💻 Author

**Mohammed Baqar**

B.Tech Information Technology Student  
Interested in Artificial Intelligence, Machine Learning and AI Engineering.

---

## ⭐ Acknowledgement

This project was created as part of learning and implementing the fundamentals of **Retrieval-Augmented Generation (RAG)** and building practical applications using Large Language Models.
