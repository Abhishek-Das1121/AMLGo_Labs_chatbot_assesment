"""
rag_pipe.py
---------------
The complete RAG pipeline  retriieval + generation combined.

What this file does:
1. Accepts a user question
2. Retrieves relevant chunks (Retriever)
3. Formats context
4. Streams a grounded answer (Generator)
5. Returns chunks + stream for the UI to consume
"""

from typing import Generator

from src.retriever import Retriever
from src.generator import Generator as AnswerGenerator



# RAG pipeline class


class RAGPipeline:
    """
    Combines the Retriever and Generator into a single pipeline.

    Used by app.py as the only import needed for the Streamlit UI.

    Example:
        pipeline = RAGPipeline()

        chunks, stream = pipeline.query(
            "What are the seller fees?"
        )

        # Show chunks in sidebar
        # Pass stream to st.write_stream()
    """

    def __init__(self):
        print("\nInitializing RAG Pipeline...")

        self.retriever = Retriever()
        self.generator = AnswerGenerator()

        print("RAG Pipeline ready.\n")

    def query(
        self,
        question: str
    ) -> tuple[list[dict], Generator[str, None, None]]:
        """
        Runs the complete RAG workflow.

        Steps:
        1. Retrieve relevant chunks
        2. Format chunks into context
        3. Generate streaming answer

        Returns:
            retrieved_chunks
            answer_stream
        """

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question)

        # Step 2: Format chunks into context
        context = self.retriever.format_context(
            retrieved_chunks
        )

        # Step 3: Create answer stream
        answer_stream = self.generator.stream_answer(
            context=context,
            question=question,
        )

        return retrieved_chunks, answer_stream




if __name__ == "__main__":

    print("\nFull RAG Pipeline Test\n")

    pipeline = RAGPipeline()

    question = "What is the eBay Money Back Guarantee?"

    print(f"Question: {question}\n")

    chunks, stream = pipeline.query(question)

    print("Retrieved chunks:\n")

    for chunk in chunks:
        print(
            f"[Chunk {chunk['chunk_id']}] "
            f"Score: {chunk['similarity_score']:.4f}"
        )

    print("\nAnswer (streaming):\n")

    for token in stream:
        print(token, end="", flush=True)

    print("\n\nPipeline test complete.\n")