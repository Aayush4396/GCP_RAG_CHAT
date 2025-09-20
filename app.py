import gradio as gr
import fitz  # PyMuPDF
import os
import sys
import numpy as np

# pip install google-genai gradio pymupdf

try:
    from google import genai
    from google.genai import types
    print("✅ google-genai imported successfully!")
except ImportError:
    print("❌ Please install: pip install google-genai gradio pymupdf")
    sys.exit(1)

# CONFIG
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    print("⚠️ Please set your Gemini API key: export GEMINI_API_KEY=your_key_here")

# Create client
client = None
if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE":
    client = genai.Client(api_key=GEMINI_API_KEY)
    print("✅ Gemini client initialized")
else:
    print("⚠️ Running without API key - features may not work")


def extract_chunks_from_uploaded_pdf(pdf_file, chunk_size=1000):
    """Extract text chunks from uploaded PDF file"""
    try:
        with fitz.open(pdf_file) as doc:
            chunks = []
            for page_num, page in enumerate(doc):
                text = page.get_text()
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size].strip()
                    if chunk and len(chunk) > 50:
                        chunks.append({
                            'text': chunk,
                            'page': page_num + 1,
                            'chunk_id': len(chunks)
                        })
        return chunks
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return []


def get_gemini_embeddings(texts):
    """Get embeddings using Gemini"""
    if not client:
        return []

    try:
        resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return [emb.values for emb in resp.embeddings]
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return []


def cosine_similarity(a, b):
    """Cosine similarity"""
    try:
        a_vec, b_vec = np.array(a), np.array(b)
        if np.linalg.norm(a_vec) == 0 or np.linalg.norm(b_vec) == 0:
            return 0
        return float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0


def find_relevant_chunks(question, chunks, top_k=3):
    """Find most relevant chunks using embeddings"""
    if not client or not chunks:
        return chunks[:top_k]

    try:
        q_resp = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[question],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        q_emb = q_resp.embeddings[0].values

        chunk_embeddings = get_gemini_embeddings([c["text"] for c in chunks])
        if not chunk_embeddings:
            return chunks[:top_k]

        sims = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
        top_idx = np.argsort(sims)[-top_k:][::-1]
        results = [chunks[i] for i in top_idx]
        for rank, idx in enumerate(top_idx):
            results[rank]['similarity'] = sims[idx]
        return results

    except Exception as e:
        print(f"Error finding relevant chunks: {e}")
        return chunks[:top_k]


def generate_answer(question, relevant_chunks):
    """Generate answer using Gemini"""
    if not client:
        return create_simple_response(relevant_chunks, question)

    try:
        context = "\n\n".join(
            f"[Page {c['page']}] {c['text']}" for c in relevant_chunks
        )

        prompt = f"""
        Based on the following document content, answer the question.
        If the answer is not in the content, say so.

        Document Content:
        {context}

        Question: {question}
        Answer:
        """

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=800
            )
        )
        return resp.text if resp.text else "❌ No response"
    except Exception as e:
        print(f"Error generating answer: {e}")
        return create_simple_response(relevant_chunks, question)


def create_simple_response(chunks, question):
    """Fallback response"""
    if not chunks:
        return "❌ No relevant content found in the document."

    resp = f"**Question:** {question}\n\n**Relevant content from your PDF:**\n\n"
    for i, c in enumerate(chunks, 1):
        sim = c.get("similarity", 0)
        resp += f"**Section {i}** (Page {c['page']}) - Relevance: {sim:.1%}\n"
        resp += f"{c['text'][:500]}{'...' if len(c['text']) > 500 else ''}\n\n"
    return resp


def rag_chat(pdf_file, question):
    """Main RAG function"""
    if not pdf_file or not question:
        return "Please upload a PDF file and enter a question."
    if not client:
        return "❌ Please configure your Gemini API key first!"

    chunks = extract_chunks_from_uploaded_pdf(pdf_file)
    if not chunks:
        return "Could not extract text from the PDF file."

    relevant = find_relevant_chunks(question, chunks)
    return generate_answer(question, relevant)


# Gradio UI
with gr.Blocks(title="Gemini RAG Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Gemini RAG PDF Chatbot")
    gr.Markdown("Upload a PDF and ask questions about it using Google's Gemini AI!")

    with gr.Row():
        api_key_input = gr.Textbox(
            label="🔑 Gemini API Key",
            placeholder="Paste your API key (get from https://aistudio.google.com/app/apikey)",
            type="password",
            value=GEMINI_API_KEY if GEMINI_API_KEY else ""
        )

        def update_key(key):
            global client, GEMINI_API_KEY
            if key:
                GEMINI_API_KEY = key
                client = genai.Client(api_key=key)
                return "✅ API key updated!"
            return "❌ Please enter a valid API key"

        key_status = gr.Textbox(label="Status", interactive=False)
        api_key_input.change(update_key, [api_key_input], [key_status])

    with gr.Row():
        with gr.Column():
            pdf_file = gr.File(label="📁 Upload PDF", file_types=[".pdf"], type="filepath")
            question_box = gr.Textbox(
                label="❓ Ask a question",
                placeholder="E.g., Summarize this report",
                lines=3
            )
            submit_btn = gr.Button("🚀 Ask", variant="primary")

        with gr.Column():
            answer_box = gr.Markdown(label="💬 Answer", height=400)

    def on_submit(pdf, q):
        if not client:
            return "❌ Please enter your Gemini API key first!"
        if pdf is None:
            return "❌ Please upload a PDF file."
        if not q.strip():
            return "❌ Please enter a question."
        return rag_chat(pdf, q.strip())

    submit_btn.click(on_submit, [pdf_file, question_box], [answer_box])
    question_box.submit(on_submit, [pdf_file, question_box], [answer_box])

# if __name__ == "__main__":
#     print("🚀 Starting Gemini RAG Chatbot...")
#     demo.launch(
#         server_name="0.0.0.0",          # Required for Cloud Run
#         server_port=int(os.getenv("PORT", 8080)),  # Cloud Run gives PORT env var
#         share=False,
#         inbrowser=False
#     )

if __name__ == "__main__":
    print("🚀 Starting Gemini RAG Chatbot...")
    demo.launch(
        server_port=8080,  # Or your preferred port
        share=False,
        inbrowser=True     # This will open the app in your browser automatically
    )

