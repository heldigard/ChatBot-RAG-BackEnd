import os
import logging
from typing import List, Dict, Any, Optional
import PyPDF2
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Clase para procesar archivos PDF y extraer texto."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Inicializa el procesador de PDFs.
        
        Args:
            chunk_size: Tamaño de los fragmentos de texto
            chunk_overlap: Superposición entre fragmentos
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extrae texto de un archivo PDF usando pdfplumber.
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Texto extraído del PDF
        """
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Página {page_num + 1} ---\n{page_text}\n"
            logger.info(f"Texto extraído exitosamente de {pdf_path}")
        except Exception as e:
            logger.error(f"Error al extraer texto de {pdf_path}: {e}")
            # Fallback a PyPDF2 si pdfplumber falla
            try:
                text = self._extract_text_with_pypdf2(pdf_path)
            except Exception as fallback_error:
                logger.error(f"Error también con PyPDF2: {fallback_error}")
                raise
        
        return text
    
    def _extract_text_with_pypdf2(self, pdf_path: str) -> str:
        """
        Método alternativo para extraer texto usando PyPDF2.
        
        Args:
            pdf_path: Ruta al archivo PDF
            
        Returns:
            Texto extraído del PDF
        """
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Página {page_num + 1} ---\n{page_text}\n"
        return text
    
    def process_pdf(self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Procesa un PDF completo y lo divide en fragmentos.
        
        Args:
            pdf_path: Ruta al archivo PDF
            metadata: Metadatos adicionales para los documentos
            
        Returns:
            Lista de documentos con fragmentos de texto
        """
        # Extraer texto del PDF
        full_text = self.extract_text_from_pdf(pdf_path)
        
        if not full_text.strip():
            logger.warning(f"No se pudo extraer texto del PDF: {pdf_path}")
            return []
        
        # Crear metadatos base
        base_metadata = {
            "source": pdf_path,
            "filename": os.path.basename(pdf_path)
        }
        if metadata:
            base_metadata.update(metadata)
        
        # Crear documento con el texto completo
        document = Document(page_content=full_text, metadata=base_metadata)
        
        # Dividir en fragmentos
        chunks = self.text_splitter.split_documents([document])
        
        # Agregar metadatos específicos para cada fragmento
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks)
            })
        
        logger.info(f"PDF procesado: {pdf_path} - {len(chunks)} fragmentos creados")
        return chunks
    
    def process_multiple_pdfs(self, pdf_directory: str) -> List[Document]:
        """
        Procesa todos los PDFs en un directorio.
        
        Args:
            pdf_directory: Ruta al directorio con PDFs
            
        Returns:
            Lista de todos los documentos procesados
        """
        all_chunks = []
        
        if not os.path.exists(pdf_directory):
            logger.error(f"El directorio no existe: {pdf_directory}")
            return all_chunks
        
        pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No se encontraron archivos PDF en: {pdf_directory}")
            return all_chunks
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_directory, pdf_file)
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error al procesar {pdf_file}: {e}")
                continue
        
        logger.info(f"Procesados {len(pdf_files)} PDFs - Total de {len(all_chunks)} fragmentos")
        return all_chunks