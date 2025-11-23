import os
import logging
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Clase para manejar embeddings y búsqueda vectorial."""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 use_openai_embeddings: bool = False,
                 openai_api_key: Optional[str] = None,
                 openai_api_base: Optional[str] = None,
                 openai_model: str = "text-embedding-ada-002"):
        """
        Inicializa el gestor de embeddings.

        Args:
            model_name: Nombre del modelo de SentenceTransformer
            use_openai_embeddings: Si es True, usa embeddings de OpenAI
            openai_api_key: API key de OpenAI
            openai_api_base: URL base de API compatible con OpenAI
            openai_model: Modelo de embeddings de OpenAI
        """
        self.use_openai_embeddings = use_openai_embeddings
        self.model_name = model_name
        self.openai_model = openai_model

        if use_openai_embeddings:
            if not openai_api_key:
                raise ValueError("Se requiere API key de OpenAI para usar embeddings de OpenAI")

            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base
            )
            logger.info(f"Usando embeddings de OpenAI con modelo: {openai_model}")
        else:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Usando SentenceTransformer con modelo: {model_name}")

        self.index = None
        self.documents = []
        self.embeddings = []

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Crea embeddings para una lista de textos.

        Args:
            texts: Lista de textos a convertir en embeddings

        Returns:
            Lista de embeddings
        """
        if self.use_openai_embeddings:
            return self._create_openai_embeddings(texts)
        else:
            return self._create_sentence_transformer_embeddings(texts)

    def _create_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Crea embeddings usando API de OpenAI."""
        embeddings = []

        # Procesar en lotes para evitar límites de API
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(
                    model=self.openai_model,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error al crear embeddings con OpenAI: {e}")
                raise

        return embeddings

    def _create_sentence_transformer_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Crea embeddings usando SentenceTransformer."""
        try:
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error al crear embeddings con SentenceTransformer: {e}")
            raise

    def build_vector_store(self, documents: List[Document]) -> None:
        """
        Construye el índice vectorial FAISS a partir de documentos.

        Args:
            documents: Lista de documentos a indexar
        """
        if not documents:
            logger.warning("No hay documentos para indexar")
            return

        self.documents = documents

        # Extraer textos de los documentos
        texts = [doc.page_content for doc in documents]

        # Crear embeddings
        logger.info(f"Creando embeddings para {len(texts)} documentos...")
        self.embeddings = self.create_embeddings(texts)

        # Crear índice FAISS
        dimension = len(self.embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)

        # Convertir a numpy array y añadir al índice
        embeddings_array = np.array(self.embeddings).astype('float32')
        self.index.add(embeddings_array)

        logger.info(f"Índice vectorial creado con {len(self.embeddings)} embeddings de dimensión {dimension}")

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """
        Busca documentos similares a la consulta.

        Args:
            query: Texto de consulta
            k: Número de resultados a retornar

        Returns:
            Lista de tuplas (documento, score)
        """
        if self.index is None:
            logger.warning("El índice vectorial no está inicializado")
            return []

        # Crear embedding para la consulta
        query_embedding = self.create_embeddings([query])[0]
        query_array = np.array([query_embedding]).astype('float32')

        # Buscar en el índice
        scores, indices = self.index.search(query_array, min(k, len(self.documents)))

        # Retornar documentos con sus scores (convertimos L2 distance a una métrica de similitud)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                # score es la distancia L2 (lower = mejor), convertimos a similitud en (0, 1]
                try:
                    l2 = float(score)
                    similarity = 1.0 / (1.0 + l2) if l2 >= 0 else 0.0
                except Exception:
                    similarity = float(score)
                results.append((self.documents[idx], float(similarity)))

        return results

    def save_vector_store(self, file_path: str) -> None:
        """
        Guarda el índice vectorial y documentos en disco.

        Args:
            file_path: Ruta base para guardar los archivos
        """
        if self.index is None:
            logger.warning("No hay índice vectorial para guardar")
            return

        try:
            # Guardar índice FAISS
            faiss.write_index(self.index, f"{file_path}.faiss")

            # Guardar documentos y embeddings
            with open(f"{file_path}.pkl", 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings,
                    'model_name': self.model_name,
                    'use_openai_embeddings': self.use_openai_embeddings,
                    'openai_model': self.openai_model
                }, f)

            logger.info(f"Vector store guardado en {file_path}")
        except Exception as e:
            logger.error(f"Error al guardar vector store: {e}")
            raise

    def load_vector_store(self, file_path: str) -> None:
        """
        Carga el índice vectorial y documentos desde disco.

        Args:
            file_path: Ruta base de los archivos guardados
        """
        try:
            # Cargar índice FAISS
            self.index = faiss.read_index(f"{file_path}.faiss")

            # Cargar documentos y metadatos
            with open(f"{file_path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.embeddings = data['embeddings']
                self.model_name = data.get('model_name', self.model_name)
                self.use_openai_embeddings = data.get('use_openai_embeddings', False)
                self.openai_model = data.get('openai_model', self.openai_model)

            # Recrear el modelo si es necesario
            if not self.use_openai_embeddings:
                self.model = SentenceTransformer(self.model_name)

            logger.info(f"Vector store cargado desde {file_path} - {len(self.documents)} documentos")
        except Exception as e:
            logger.error(f"Error al cargar vector store: {e}")
            raise

    def get_document_count(self) -> int:
        """Retorna el número de documentos en el índice."""
        return len(self.documents) if self.documents else 0
