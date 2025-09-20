***

# GCP_RAG_CHAT

**GCP_RAG_CHAT** is a simple, production-ready Retrieval-Augmented Generation (RAG) chatbot powered by Google's Gemini AI. Upload any PDF, ask questions about its contents, and get instant, context-aware answers. Deployable on Google Cloud Run with a single Docker command.

## Features

- **PDF Ingestion:** Securely upload your PDF documents for Q\&A.
- **RAG Architecture:** Retrieves and selects the most relevant pieces of your PDF using Gemini embeddings.
- **LLM-Powered Answers:** Uses Gemini AI to generate context-based responses.
- **Simple UI:** Built with Gradio for a fast and intuitive user experience.
- **Cloud Native:** Easily containerized and ready for GCP Cloud Run deployment.
- **Extensible:** Easily adaptable for other data sources or LLM APIs.


## Project Structure

- `app.py`: Main application code. Handles PDF ingestion, chunking, embedding with Gemini, retrieval, and the chat interface via Gradio.
- `requirements.txt`: Lists dependencies (`gradio`, `pymupdf`, `numpy`, `google-genai`).
- `Dockerfile`: Containerizes the app for simple deployment.
- `rag_flow.html`: Visual diagram of the RAG pipeline.
- `Chatbot_Documentation.pdf` \& `RAG_Documentation.docx`: Project documentation and sample use cases.


## Quick Start

1. **Clone the Repository**

```bash
git clone https://github.com/Aayush4396/GCP_RAG_CHAT.git
cd GCP_RAG_CHAT
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Set your Gemini API Key**
    - Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)
    - Export it:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

4. **Run Locally**

```bash
python app.py
```

Visit `http://localhost:8080` to access the chatbot.
5. **Docker Deployment**

```bash
docker build -t gcp-rag-chat .
docker run -p 8080:8080 -e GEMINI_API_KEY='your-api-key-here' gcp-rag-chat
```

6. **GCP Cloud Run**
    - Once containerized, push the image to Google Container Registry and deploy on Cloud Run.

## How it Works

1. **Upload a PDF:** The bot extracts the text and splits it into manageable chunks.
2. **Embeddings:** Gemini creates vector embeddings for all chunks.
3. **Retrieval:** On receiving a user query, the bot computes its embedding and compares with document chunks using cosine similarity.
4. **Generation:** The most relevant chunks are provided as context to Gemini's LLM for accurate answer synthesis.

## Requirements

- Python 3.11+
- GEMINI_API_KEY from Google Cloud AI Studio


## Credits

- Built with [Gradio](https://www.gradio.app/), [Google Gemini AI](https://ai.google.dev/), and [PyMuPDF](https://github.com/pymupdf/PyMuPDF).

***

<span style="display:none">[^1][^2]</span>

