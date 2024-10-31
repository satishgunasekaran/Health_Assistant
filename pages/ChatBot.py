import streamlit as st
import sys
from typing import List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from src.services.rag_service import RagService
from src.services.chat_service import ChatService

sys.path.append("")  # Adjust if necessary to include your module path

class PersistentChatService:
    @staticmethod
    def initialize_services():
        if "rag_service" not in st.session_state:
            st.session_state.rag_service = RagService()
        if "chat_service" not in st.session_state:
            st.session_state.chat_service = ChatService(
                model_name="meta-llama/Llama-3-70b-chat-hf",
                rag_service=st.session_state.rag_service
            )
            st.session_state.chat_service.history = []
        return st.session_state.chat_service, st.session_state.rag_service

    @staticmethod
    def get_history() -> List:
        if "chat_service" not in st.session_state:
            return []
        return st.session_state.chat_service.history

# Page configuration
st.set_page_config(page_title="Health Assistant Chatbot", page_icon="💬", layout="wide")
st.title("💬 Health Assistant Chatbot")

# Initialize the services
chat_service, rag_service = PersistentChatService.initialize_services()

# Sidebar for document management
with st.sidebar:
    st.header("📄 Document Management")
    
    # File upload section
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing PDF..."):
                file_id = rag_service.process_pdf(uploaded_file)
                st.success(f"Successfully processed: {uploaded_file.name}")
    
    # Display current files and management options
    st.subheader("Uploaded Documents")
    files = rag_service.get_files_list()
    
    if not files:
        st.info("No documents uploaded")
    else:
        for file in files:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📑 {file['name']}")
                st.caption(f"Chunks: {file['num_chunks']} | Uploaded: {file['upload_time'].strftime('%H:%M:%S')}")
            with col2:
                if st.button("🗑️", key=f"delete_{file['id']}", help="Delete this document"):
                    rag_service.delete_file(file['id'])
                    st.rerun()
    
    # Option to clear all documents
    if files:
        if st.button("Clear All Documents"):
            rag_service.clear_all_contexts()
            st.success("All documents cleared")
            st.rerun()

# Initialize messages in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message("👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"], unsafe_allow_html=True)

# React to user input
if prompt := st.chat_input("💡 Ask your health question or inquire about the uploaded documents..."):
    # Display user message
    with st.chat_message("👤"):
        st.markdown(f"**{prompt}**")
    
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": f"**{prompt}**"})
    
    # Prepare a container for the assistant's streaming response
    assistant_response = st.chat_message("🤖")
    
    # Stream the assistant's response
    response_stream = chat_service.stream_chat(prompt)
    full_response = assistant_response.write_stream(response_stream)
    
    # Add assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Debug information in sidebar
with st.sidebar:
    st.write("---")
    st.write("Debug Information:")
    st.write(f"Chat history length: {len(PersistentChatService.get_history())}")
    st.write(f"Active documents: {len(rag_service.get_files_list())}")

if __name__ == "__main__":
    pass  # The Streamlit server will handle running the app