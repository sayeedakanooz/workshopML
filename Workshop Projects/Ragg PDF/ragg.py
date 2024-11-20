import streamlit as st
import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoderTokenizer, BartForConditionalGeneration, BartTokenizer
import faiss
import numpy as np
import pdfplumber

# Function to load DPR models and tokenizers from Hugging Face
def load_dpr_models():
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    return question_encoder, context_encoder, question_tokenizer, context_tokenizer

# Function to load BART for text generation
def load_bart_model():
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    return model, tokenizer

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to encode context and query using DPR model
def encode_dpr(texts, encoder, tokenizer, max_length=512):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        embeddings = encoder(**inputs).pooler_output
    return embeddings

# Function to build FAISS index
def build_faiss_index(documents, context_encoder, context_tokenizer):
    # Encode the documents into dense vectors using DPR Context Encoder
    embeddings = encode_dpr(documents, context_encoder, context_tokenizer)
    embeddings = embeddings.cpu().numpy()

    # Build FAISS index
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings)
    return faiss_index

# Function to retrieve the most relevant documents
def retrieve_documents(query, faiss_index, documents, question_encoder, question_tokenizer, k=3):
    query_embedding = encode_dpr([query], question_encoder, question_tokenizer)
    query_embedding = query_embedding.cpu().numpy()
    
    # Perform the retrieval using FAISS
    distances, indices = faiss_index.search(query_embedding, k)
    relevant_documents = [documents[i] for i in indices[0]]
    return relevant_documents

# Function to generate answer using BART
def generate_answer(question, context, model, tokenizer):
    input_text = f"question: {question}  context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True, padding=True)
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, early_stopping=True)
    answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return answer

# Streamlit UI
def main():
    st.title("RAG-based PDF Question Answering with Hugging Face")
    st.write("Upload a PDF file and ask a question. The system retrieves relevant documents and generates an answer.")

    # Upload PDF
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

    # Initialize models only once when the user uploads a PDF
    if pdf_file is not None:
        # Extract text from PDF
        document_text = extract_text_from_pdf(pdf_file)
        st.write("Extracted Text from PDF (first 500 characters):")
        st.write(document_text[:500])  # Display preview of the extracted text

        # Split the document into paragraphs or sentences (chunks)
        documents = document_text.split("\n\n")

        # Load models if they are not already loaded
        if 'models' not in st.session_state:
            question_encoder, context_encoder, question_tokenizer, context_tokenizer = load_dpr_models()
            bart_model, bart_tokenizer = load_bart_model()
            st.session_state.models = {
                "question_encoder": question_encoder,
                "context_encoder": context_encoder,
                "question_tokenizer": question_tokenizer,
                "context_tokenizer": context_tokenizer,
                "bart_model": bart_model,
                "bart_tokenizer": bart_tokenizer
            }

        # Build the FAISS index
        if 'faiss_index' not in st.session_state:
            faiss_index = build_faiss_index(documents, st.session_state.models["context_encoder"], st.session_state.models["context_tokenizer"])
            st.session_state.faiss_index = faiss_index
            st.session_state.documents = documents  # Save documents for later retrieval

        # Ask a question
        question = st.text_input("Ask a question:")

        if question:
            # Retrieve relevant documents from the index
            relevant_documents = retrieve_documents(
                question, 
                st.session_state.faiss_index, 
                st.session_state.documents, 
                st.session_state.models["question_encoder"], 
                st.session_state.models["question_tokenizer"], 
                k=3
            )
            context = " ".join(relevant_documents)

            # Generate an answer based on the context and the question
            answer = generate_answer(question, context, st.session_state.models["bart_model"], st.session_state.models["bart_tokenizer"])

            # Display the generated answer
            st.write("Generated Answer:")
            st.write(answer)

if __name__ == "__main__":
    main()


pip install streamlit faiss-cpu transformers torch pdfplumber
