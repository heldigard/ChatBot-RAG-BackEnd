import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

def load_system_prompt() -> str:
    """Carga el SystemPrompt desde el archivo."""
    try:
        with open('SystemPrompt.txt', 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        logger.error(f"Error al cargar SystemPrompt.txt: {e}")
        # Fallback a un prompt básico
        return """
Eres un asistente legal experto. Responde basándote únicamente en la información proporcionada.
Si no encuentras la respuesta en los documentos, indica que no tienes esa información.
"""

class LLMManager:
    """Clase para manejar la interacción con APIs compatibles con OpenAI."""

    def __init__(self,
                 api_key: str,
                 api_base: Optional[str] = None,
                 model: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 max_tokens: int = 1000):
        """
        Inicializa el gestor de LLM.

        Args:
            api_key: API key para el servicio
            api_base: URL base de API compatible con OpenAI (opcional)
            model: Modelo a usar
            temperature: Temperatura para generación
            max_tokens: Máximo número de tokens
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Configurar cliente OpenAI
        client_kwargs = {"api_key": api_key}
        if api_base:
            client_kwargs["base_url"] = api_base

        logger.debug("LLMManager: Inicializando cliente (api_key masked) base_url=%s", api_base)
        self.client = OpenAI(**client_kwargs)

        logger.info(f"LLM Manager inicializado con modelo: {model}")

    def generate_response(self,
                         query: str,
                         context: str,
                         system_prompt: Optional[str] = None,
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Genera una respuesta usando el LLM con contexto RAG.

        Args:
            query: Pregunta del usuario
            context: Contexto recuperado de los documentos
            system_prompt: Prompt del sistema (opcional)
            conversation_history: Historial de conversación (opcional)

        Returns:
            Diccionario con respuesta y metadatos
        """
        # Construir mensajes
        messages = []

        # Añadir system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Usar el SystemPrompt personalizado del archivo
            default_system_prompt = load_system_prompt()
            messages.append({"role": "system", "content": default_system_prompt})

        # Añadir historial de conversación si existe
        if conversation_history:
            messages.extend(conversation_history)

        # Construir el prompt con contexto y pregunta
        user_prompt = f"""
Contexto:
{context}

Pregunta: {query}

Responde basándote en el contexto proporcionado. Sé específico y menciona las fuentes cuando sea posible.
"""
        messages.append({"role": "user", "content": user_prompt})

        try:
            # Realizar la llamada a la API
            logger.debug("LLMManager: Llamando a API con model=%s, base_url=%s", self.model, self.client.base_url)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Extraer respuesta
            answer = response.choices[0].message.content

            # Extraer metadatos
            usage = response.usage
            metadata = {
                "model": self.model,
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
                "finish_reason": response.choices[0].finish_reason
            }

            return {
                "answer": answer,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error al generar respuesta con LLM: {e}")
            return {
                "answer": f"Error al generar respuesta: {str(e)}",
                "metadata": {"error": str(e)}
            }

    def generate_streaming_response(self,
                                   query: str,
                                   context: str,
                                   system_prompt: Optional[str] = None,
                                   conversation_history: Optional[List[Dict[str, str]]] = None):
        """
        Genera una respuesta en streaming usando el LLM.

        Args:
            query: Pregunta del usuario
            context: Contexto recuperado de los documentos
            system_prompt: Prompt del sistema (opcional)
            conversation_history: Historial de conversación (opcional)

        Yields:
            Trozos de la respuesta generada
        """
        # Construir mensajes (igual que en generate_response)
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        else:
            # Usar el SystemPrompt personalizado del archivo
            default_system_prompt = load_system_prompt()
            messages.append({"role": "system", "content": default_system_prompt})

        if conversation_history:
            messages.extend(conversation_history)

        user_prompt = f"""
Contexto:
{context}

Pregunta: {query}

Responde basándote en el contexto proporcionado. Sé específico y menciona las fuentes cuando sea posible.
"""
        messages.append({"role": "user", "content": user_prompt})

        try:
            # Realizar la llamada en streaming
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True
            )

            # Yield cada trozo de la respuesta
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            logger.error(f"Error en streaming con LLM: {e}")
            yield f"Error al generar respuesta: {str(e)}"

    def test_connection(self) -> bool:
        """
        Prueba la conexión con la API.

        Returns:
            True si la conexión es exitosa, False en caso contrario
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hola"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            logger.error(f"Error al probar conexión: {e}")
            return False
