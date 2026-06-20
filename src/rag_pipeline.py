"""
rag_pipeline.py
---------------
The single orchestrates that app.py talks to.

Responsibilities:
- Index PDFs only once.
- Handle retrieval + generation.
- Keep app.py thin and simple.
"""

from src.pdf_preprocess import process_pdf
from src.chunking import chunk_pages
from src.vector_store import collection_exists, build_collection
from src.retriever import retrieve
from src.generator import stream_answer
from src.utils import get_recent_history


class RAGPipeline:
    """
    Main orchestrator used by app.py.

    Example:

        pipeline = RAGPipeline()

        result = pipeline.index_document(file_bytes, filename)

        chunks, token_stream = pipeline.ask(
            pdf_hash,
            question,
            messages
        )
    """

    def index_document(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> dict:
        """
        Process and index a PDF only once.

        Returns:
        {
            "pdf_hash": str,
            "filename": str,
            "saved_path": Path,
            "total_pages": int,
            "total_chunks": int,
            "was_already_indexed": bool,
        }
        """

        # Process PDF
        result = process_pdf(
            file_bytes=file_bytes,
            filename=filename,
        )

        pdf_hash = result["pdf_hash"]

        # Skip if already indexed
        if collection_exists(pdf_hash):

            print(
                f"[pipeline] '{filename}' already indexed "
                f"(hash={pdf_hash[:8]}...)."
            )

            return {
                **result,
                "total_chunks": 0,
                "was_already_indexed": True,
            }

        # Create chunks
        chunks = chunk_pages(
            result["pages"],
            source_filename=filename,
        )

        # Build vector DB
        build_collection(
            pdf_hash=pdf_hash,
            chunks=chunks,
        )

        return {
            **result,
            "total_chunks": len(chunks),
            "was_already_indexed": False,
        }

    def ask(
        self,
        pdf_hash: str,
        question: str,
        messages: list[dict] | None = None,
        k: int = 5,
    ):
        """
        Retrieval + generation.

        Returns:
            chunks
            token_stream
        """

        if not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        history = get_recent_history(
            messages or []
        )

        chunks = retrieve(
            pdf_hash=pdf_hash,
            query=question,
            k=k,
        )

        if len(chunks) == 0:
            raise RuntimeError(
                "No chunks retrieved."
            )

        token_stream = stream_answer(
            question=question,
            chunks=chunks,
            history=history,
        )

        return chunks, token_stream