from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from typing import List, Dict
from datetime import datetime
import tempfile
import os

class RagService:
    """Service for handling document processing and retrieval."""
    
    def __init__(self):
        self.document_contexts: Dict[str, List[Document]] = {}  # Store contexts by file ID
        self.files_info: Dict[str, dict] = {}  # Store file metadata
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def process_pdf(self, pdf_file) -> str:
        """Process a PDF file and return file ID."""
        # Generate a unique file ID
        file_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{pdf_file.name}"
        
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_path = tmp_file.name

        try:
            # Load and process the PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            
            # Split the documents into chunks
            chunks = self.text_splitter.split_documents(pages)
            
            # Store the chunks and file info
            self.document_contexts[file_id] = chunks
            self.files_info[file_id] = {
                'name': pdf_file.name,
                'upload_time': datetime.now(),
                'num_chunks': len(chunks)
            }
            
            return file_id
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)

    def delete_file(self, file_id: str) -> bool:
        """Delete a file and its context."""
        if file_id in self.document_contexts:
            del self.document_contexts[file_id]
            del self.files_info[file_id]
            return True
        return False

    def get_files_list(self) -> List[dict]:
        """Get list of all processed files with their info."""
        return [
            {'id': file_id, **info}
            for file_id, info in self.files_info.items()
        ]

    def get_relevant_context(self, query: str, num_chunks: int = 3) -> str:
        """Get the most relevant context chunks from all loaded documents."""
        if not self.document_contexts:
            return ""
        
        def score_chunk(chunk: Document, query: str) -> float:
            query_words = set(query.lower().split())
            chunk_words = set(chunk.page_content.lower().split())
            return len(query_words.intersection(chunk_words))
        
        # Score chunks from all documents
        all_scored_chunks = []
        for chunks in self.document_contexts.values():
            scored_chunks = [(chunk, score_chunk(chunk, query)) for chunk in chunks]
            all_scored_chunks.extend(scored_chunks)
        
        # Sort and get top chunks
        all_scored_chunks.sort(key=lambda x: x[1], reverse=True)
        relevant_chunks = [chunk.page_content for chunk, _ in all_scored_chunks[:num_chunks]]
        
        return "\n\n".join(relevant_chunks)

    def clear_all_contexts(self):
        """Clear all document contexts."""
        self.document_contexts.clear()
        self.files_info.clear()