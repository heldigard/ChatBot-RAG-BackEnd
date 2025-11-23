import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from pdf_processor import PDFProcessor
from embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

class RAGSystem:
    """Sistema completo de Retrieval-Augmented Generation."""

    def __init__(self,
                 pdf_directory: str = "./pdfs",
                 vector_store_path: str = "./vector_store",
                 use_openai_embeddings: bool = False,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: Optional[str] = None,
                 openai_embedding_api_base: Optional[str] = None,
                 openai_embedding_api_key: Optional[str] = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 openai_embedding_model: str = "text-embedding-ada-002",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Inicializa el sistema RAG.

        Args:
            pdf_directory: Directorio con archivos PDF
            vector_store_path: Ruta para guardar/cargar el vector store
            use_openai_embeddings: Si es True, usa embeddings de OpenAI
            openai_api_key: API key de OpenAI
            openai_api_base: URL base de API compatible con OpenAI
            openai_embedding_api_base: URL base específica para API de embeddings (opcional)
            openai_embedding_api_key: API key específica para embeddings (opcional)
            embedding_model: Modelo de embeddings
            openai_embedding_model: Modelo de embeddings de OpenAI/OpenRouter
            chunk_size: Tamaño de los fragmentos de texto
            chunk_overlap: Superposición entre fragmentos
        """
        self.pdf_directory = pdf_directory
        self.vector_store_path = vector_store_path

        # Inicializar procesador de PDFs
        self.pdf_processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Inicializar gestor de embeddings
        self.embedding_manager = EmbeddingManager(
            model_name=embedding_model,
            use_openai_embeddings=use_openai_embeddings,
            openai_api_key=openai_embedding_api_key or openai_api_key,
            openai_api_base=openai_embedding_api_base or openai_api_base,
            openai_model=openai_embedding_model
        )

        # Intentar cargar vector store existente
        self._load_or_create_vector_store()

    def _load_or_create_vector_store(self) -> None:
        """Carga un vector store existente o crea uno nuevo."""
        if os.path.exists(f"{self.vector_store_path}.faiss") and \
           os.path.exists(f"{self.vector_store_path}.pkl"):
            try:
                self.embedding_manager.load_vector_store(self.vector_store_path)
                logger.info("Vector store cargado exitosamente")
            except Exception as e:
                logger.error(f"Error al cargar vector store: {e}")
                self._create_new_vector_store()
        else:
            self._create_new_vector_store()

    def _create_new_vector_store(self) -> None:
        """Crea un nuevo vector store a partir de los PDFs."""
        logger.info("Creando nuevo vector store...")

        # Procesar todos los PDFs
        documents = self.pdf_processor.process_multiple_pdfs(self.pdf_directory)

        if not documents:
            logger.warning("No se encontraron documentos para procesar")
            return

        # Construir índice vectorial
        self.embedding_manager.build_vector_store(documents)

        # Guardar vector store
        self.embedding_manager.save_vector_store(self.vector_store_path)

        logger.info(f"Vector store creado con {len(documents)} documentos")

    def retrieve_documents(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Recupera documentos relevantes para una consulta.

        Args:
            query: Consulta de búsqueda
            k: Número de documentos a recuperar

        Returns:
            Lista de tuplas (documento, score)
        """
        return self.embedding_manager.search(query, k)

    def format_context(self, retrieved_docs: List[Tuple[Document, float]], max_context_chars: Optional[int] = None) -> str:
        """
        Formatea los documentos recuperados como contexto.

        Args:
            retrieved_docs: Documentos recuperados con sus scores
            max_context_chars: Límite máximo de caracteres para el contexto

        Returns:
            Contexto formateado como string
        """
        if not retrieved_docs:
            return "No se encontró información relevante en los documentos."

        context_parts = []
        doc_counters = {}  # Para llevar contador de fragmentos por documento

        for doc, score in retrieved_docs:
            metadata = doc.metadata
            filename = metadata.get('filename', 'Desconocido')
            chunk_id = metadata.get('chunk_id', 0)

            # Incrementar contador para este documento
            if filename not in doc_counters:
                doc_counters[filename] = 0
            doc_counters[filename] += 1

            context_part = f"""
{filename} (Fragmento: {doc_counters[filename]}, Relevancia: {score:.4f}):
{doc.page_content}
"""
            context_parts.append(context_part)

        full_context = "\n".join(context_parts)
        if max_context_chars and len(full_context) > max_context_chars:
            # Truncate to max_context_chars while keeping whole document parts
            trimmed = ""
            for part in context_parts:
                if len(trimmed) + len(part) > max_context_chars:
                    break
                trimmed += part
            trimmed += "\n... (Contexto truncado)"
            return trimmed

        return full_context

    def get_sources_info(self, retrieved_docs: List[Tuple[Document, float]]) -> List[Dict[str, Any]]:
        """
        Extrae información de fuentes de los documentos recuperados.

        Args:
            retrieved_docs: Documentos recuperados con sus scores

        Returns:
            Lista de información de fuentes
        """
        sources = []
        seen_sources = set()

        for doc, score in retrieved_docs:
            metadata = doc.metadata
            source = metadata.get('filename', 'Desconocido')

            if source not in seen_sources:
                sources.append({
                    'filename': source,
                    'source': metadata.get('source', ''),
                    'score': score
                })
                seen_sources.add(source)

        return sources

    def rebuild_vector_store(self) -> None:
        """Reconstruye completamente el vector store."""
        logger.info("Reconstruyendo vector store...")

        # Eliminar archivos existentes
        for ext in ['.faiss', '.pkl']:
            file_path = f"{self.vector_store_path}{ext}"
            if os.path.exists(file_path):
                os.remove(file_path)

        # Crear nuevo vector store
        self._create_new_vector_store()

    def add_documents(self, documents: List[Document]) -> None:
        """
        Añade nuevos documentos al vector store.

        Args:
            documents: Lista de documentos a añadir
        """
        if not documents:
            return

        # Obtener documentos existentes
        existing_docs = self.embedding_manager.documents

        # Añadir nuevos documentos
        all_docs = existing_docs + documents

        # Reconstruir vector store
        self.embedding_manager.build_vector_store(all_docs)
        self.embedding_manager.save_vector_store(self.vector_store_path)

        logger.info(f"Añadidos {len(documents)} nuevos documentos al vector store")

    def get_stats(self) -> Dict[str, Any]:
        """
        Retorna estadísticas del sistema RAG.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'total_documents': self.embedding_manager.get_document_count(),
            'pdf_directory': self.pdf_directory,
            'vector_store_path': self.vector_store_path,
            'embedding_model': self.embedding_manager.model_name,
            'use_openai_embeddings': self.embedding_manager.use_openai_embeddings
        }
