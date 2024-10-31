from langchain_together import ChatTogether
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from typing import List, Generator
from dotenv import load_dotenv
from .rag_service import RagService

load_dotenv()

class ChatService:
    """Service for handling conversations with context from RAG."""
    
    def __init__(self, model_name: str, rag_service: RagService):
        self.llm = ChatTogether(
            model=model_name,
            temperature=0.5,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.rag_service = rag_service
        self.history = []

    def get_messages_with_context(self, user_input: str) -> List:
        """Combine system message, context, and chat history."""
        system_message = SystemMessage(content="You are a helpful health assistant. Answer questions about general health, wellness, and lifestyle.")
        
        context = self.rag_service.get_relevant_context(user_input)
        if context:
            context_message = SystemMessage(content=f"Reference the following information when relevant:\n\n{context}")
            return [system_message, context_message] + self.history
        
        return [system_message] + self.history

    def chat(self, user_input: str) -> str:
        """Regular chat method with context awareness."""
        self.history.append(HumanMessage(content=user_input))
        messages = self.get_messages_with_context(user_input)
        
        response = self.llm.invoke(messages)
        self.history.append(AIMessage(content=response.content))
        
        return response.content

    def stream_chat(self, user_input: str) -> Generator[str, None, None]:
        """Streaming chat method with context awareness."""
        self.history.append(HumanMessage(content=user_input))
        messages = self.get_messages_with_context(user_input)
        
        full_response = ""
        for response_chunk in self.llm.stream(messages):
            chunk_content = getattr(response_chunk, 'content', str(response_chunk))
            yield chunk_content
            full_response += chunk_content
        
        self.history.append(AIMessage(content=full_response))

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()