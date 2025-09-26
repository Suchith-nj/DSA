import os
import openai
import PyPDF2
import streamlit as st
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import json

class chatbotRAG:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.embedding_model = "text-embedding-3-small"
        
        # Text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Storage for chunks and embeddings
        self.chunks = []
        self.embeddings = []
        self.document_name = ""
    
    def extract_text_from_pdf(self, file_bytes):
        """Extract text from PDF"""
        try:
            reader = PyPDF2.PdfReader(file_bytes)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def create_embeddings(self, texts):
        """Create embeddings for text chunks"""
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            embeddings.append(response.data[0].embedding)
        return embeddings
    
    def process_document(self, file_bytes, filename):
        """Process PDF and create vector database"""
        # Extract text
        text = self.extract_text_from_pdf(file_bytes)
        
        if not text.strip():
            raise ValueError("No text found in PDF")
        
        # Split into chunks
        self.chunks = self.text_splitter.split_text(text)
        
        # Create embeddings
        self.embeddings = self.create_embeddings(self.chunks)
        self.document_name = filename
        
        return len(self.chunks)
    
    def find_relevant_chunks(self, query, top_k=3):
        """Find most relevant chunks for the query"""
        if not self.chunks:
            return []
        
        # Get query embedding
        query_response = self.client.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        query_embedding = query_response.data[0].embedding
        
        # Calculate similarities
        query_embedding = np.array(query_embedding).reshape(1, -1)
        chunk_embeddings = np.array(self.embeddings)
        
        similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Get top k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_chunks = []
        for idx in top_indices:
            relevant_chunks.append({
                'text': self.chunks[idx],
                'similarity': similarities[idx],
                'chunk_id': idx
            })
        
        return relevant_chunks
    
    def generate_response(self, query, context_chunks):
        """Generate response using LLM with context"""
        if not context_chunks:
            return "I don't have enough information to answer your question. Please make sure you've uploaded a document."
        
        # Prepare context from relevant chunks
        context = "\n\n".join([chunk['text'] for chunk in context_chunks])
        
        # Create prompt
        prompt = f"""Based on the following context from the document, please answer the question. If the answer is not in the context, say so.

            Context:
            {context}

            Question: {query}

            Answer:
            
            """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be accurate and cite the context when possible."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def chat(self, query):
        """Main chat function"""
        # Find relevant chunks
        relevant_chunks = self.find_relevant_chunks(query, top_k=3)
        
        # Generate response
        response = self.generate_response(query, relevant_chunks)
        
        return {
            'response': response,
            'sources': relevant_chunks
        }
    
    def save_rag_data(self, filepath):
        """Save RAG data to file for future use"""
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings,
            'document_name': self.document_name
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_rag_data(self, filepath):
        """Load RAG data from file for future use"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.embeddings = data['embeddings']
        self.document_name = data['document_name']





def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title(" RAG Chatbot - Chat with Your PDF")
    st.write("Upload a PDF document and chat with its content using AI")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        
        model = st.selectbox(
            "Model",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        )
        
        st.header(" Document Upload")
        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=['pdf']
        )
        
        if uploaded_file and api_key:
            if st.button("Process Document", type="primary"):
                try:
                    # Initialize RAG system
                    st.session_state.rag_system = chatbotRAG(api_key, model)
                    
                    with st.spinner("Processing document..."):
                        file_bytes = BytesIO(uploaded_file.read())
                        chunk_count = st.session_state.rag_system.process_document(
                            file_bytes, 
                            uploaded_file.name
                        )
                    
                    st.session_state.document_processed = True
                    st.success(f"Document processed! Created {chunk_count} chunks.")
                    
                    # Clear previous chat history
                    st.session_state.chat_history = []
                    
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        
        # Document info
        if st.session_state.document_processed and st.session_state.rag_system:
            st.info(f"Loaded: {st.session_state.rag_system.document_name}")
            st.info(f"Chunks: {len(st.session_state.rag_system.chunks)}")
    
    # Main chat interface
    if not api_key:
        st.warning(" Please enter your OpenAI API key in the sidebar to get started.")
        return
    
    if not st.session_state.document_processed:
        st.info("Please upload and process a PDF document to start chatting.")
        return
    
    # Chat interface
    st.header("Chat with your document")
    
    # Display chat history, Just looping all qsns and answers into container
    chat_container = st.container()
    with chat_container:
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            # User message
            st.chat_message("user").write(question)
            
            # Assistant response
            with st.chat_message("assistant"):
                st.write(answer)
                
                # Show sources
                with st.expander(" Sources"):
                    for j, source in enumerate(sources):
                        st.write(f"**Chunk {source['chunk_id']} (Similarity: {source['similarity']:.3f})**")
                        st.write(source['text'][:200] + "..." if len(source['text']) > 200 else source['text'])
                        st.divider()
    
    # Chat input
    query = st.chat_input("Ask a question about your document...")
    
    if query:
        try:
            # Add user message to chat
            with st.chat_message("user"):
                st.write(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.rag_system.chat(query)
                
                st.write(result['response'])
                
                # Show sources
                with st.expander(" Sources"):
                    for i, source in enumerate(result['sources']):
                        st.write(f"**Chunk {source['chunk_id']} (Similarity: {source['similarity']:.3f})**")
                        st.write(source['text'][:200] + "..." if len(source['text']) > 200 else source['text'])
                        if i < len(result['sources']) - 1:
                            st.divider()
            
            # Add to chat history
            st.session_state.chat_history.append((
                query, 
                result['response'], 
                result['sources']
            ))
            
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
    
    # Clear chat button
    if st.session_state.chat_history:
        if st.button(" Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()