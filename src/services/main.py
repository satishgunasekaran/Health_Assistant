import sys
sys.path.append("")  # Adjust if necessary to include your module path

from src.services.rag_service import RagService
from src.services.chat_service import ChatService

def main():
    # Initialize services
    rag_service = RagService()
    chat_service = ChatService(
        model_name="meta-llama/Llama-3-70b-chat-hf",
        rag_service=rag_service
    )
    
    # Example: Process a PDF file
    with open("data/Sample-filled-in-MR.pdf", "rb") as pdf_file:
        file_id = rag_service.process_pdf(pdf_file)
        print(f"Processed file with ID: {file_id}")
    
    # Show list of processed files
    files = rag_service.get_files_list()
    print("\nProcessed files:")
    for file in files:
        print(f"- {file['name']} (ID: {file['id']})")
    
    # Example chat interaction
    questions = [
        "What is the patient's name?",
        "What medications are they taking?",
        "What is their medical history?"
    ]
    
    print("\nChat interactions:")
    for question in questions:
        print(f"\nQ: {question}")
        response = chat_service.chat(question)
        print(f"A: {response}")
    
    # Example of streaming response
    print("\nStreaming response example:")
    print("Q: Can you summarize the patient's condition?")
    print("A: ", end="", flush=True)
    for chunk in chat_service.stream_chat("Can you summarize the patient's condition?"):
        print(chunk, end="", flush=True)
    print()

if __name__ == "__main__":
    main()