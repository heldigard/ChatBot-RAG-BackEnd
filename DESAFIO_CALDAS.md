**HACKATHON CALDAS 2025**

**HACK-KOGNIA 1.0: ASISTENTE LEGAL INTELIGENTE CON IA**

**NIVEL INTERMEDIO**

**RESUMEN**

KOGNIA IA presenta el reto Hack-Kognia 1.0, cuyo propósito es construir
un asistente legal basado en inteligencia artificial capaz de
interpretar documentos jurídicos y responder preguntas sobre su
contenido mediante un chat interactivo.

El asistente no redacta textos nuevos: analiza la información existente
en los documentos cargados por el usuario (contratos, estatutos,
acuerdos, etc.) y genera respuestas contextualizadas y precisas.

Este desafío propone aplicar técnicas de Recuperación Aumentada por
Generación (RAG) y modelos de lenguaje (LLM) para transformar la forma
en que se accede a la información legal, facilitando la comprensión y
búsqueda de contenido jurídico por parte de cualquier usuario.

**OBJETIVO DEL RETO**

Desarrollar un prototipo funcional (MVP) de un asistente legal
inteligente, que permita cargar uno o varios documentos legales y
realizar preguntas en lenguaje natural, obteniendo respuestas
fundamentadas en los textos cargados.

El sistema deberá operar desde una URL pública, con interfaz clara y
usabilidad básica, demostrando la integración real de componentes de
procesamiento, consulta semántica y generación de respuestas.

**NIVEL INTERMEDIO -- ASISTENTE LEGAL CON IA**

Este reto está dirigido a equipos que ya poseen fundamentos en
desarrollo web, manejo de datos y aprendizaje automático, y que desean
implementar tecnologías de lenguaje natural aplicadas (NLP, RAG y LLMs).

Los participantes deberán combinar sus habilidades en backend, IA y
frontend para construir una solución práctica y escalable, orientada al
campo jurídico.

**CONTEXTO DEL PROBLEMA**

El volumen de documentos legales en organizaciones, empresas y entidades
públicas crece constantemente, haciendo cada vez más compleja la
búsqueda de información relevante.\
La mayoría de usuarios carecen de herramientas que les permitan
consultar, interpretar y comprender textos jurídicos extensos de forma
intuitiva.

El reto busca demostrar cómo un modelo de IA bien entrenado y un sistema
de búsqueda semántica pueden democratizar el acceso a la información
legal, reduciendo tiempos de análisis y errores interpretativos.\
El desafío, por tanto, combina tecnología, lenguaje y derecho, aplicando
inteligencia artificial al servicio de la transparencia y la
accesibilidad.

**OBJETIVO ESPECÍFICO**

Construir un sistema end-to-end con tres componentes principales:

1.  **Carga e indexación de documentos**

- Permitir al usuario subir uno o varios archivos en formato PDF o
  texto.

- Dividir e indexar los documentos para su análisis mediante embeddings.

2.  **Módulo de búsqueda y razonamiento (RAG)**

- Recuperar fragmentos relevantes del documento ante una consulta.

- Generar respuestas precisas, fundamentadas en los textos cargados.

3.  **Interfaz tipo chat**

- Permitir interacción fluida con el usuario.

- Mostrar respuestas, fragmentos fuente y nivel de confianza.

**REQUISITOS TÉCNICOS**

**Frameworks sugeridos**

- Procesamiento y RAG: LangChain, LangGraph o LlamaIndex.

- Modelos LLM: OpenAI (GPT-4 / 4o), Claude, Gemini o Mistral.

- Bases vectoriales: FAISS, ChromaDB o Pinecone.

- Backend: Python (FastAPI / Flask) o Node.js (Express).

- Frontend: React + Tailwind CSS / Next.js / JavaScript puro.

- Base de datos opcional: PostgreSQL, Supabase o MongoDB.

**RESULTADO ESPERADO**

Un asistente legal en línea que:

1.  Permita cargar documentos y hacer consultas en lenguaje natural.

2.  Ofrezca respuestas basadas en los textos cargados (sin invención).

3.  Sea accesible desde una URL pública y muestre una demo funcional.

4.  Cuente con un repositorio organizado, documentación técnica y pitch
    demostrativo.

**RECOMENDACIONES**

- Centren el tiempo de desarrollo en lograr la integración funcional RAG
  antes que en la estética.

- Documenten los pasos de indexación y embeddings en el notebook.

- Incluyan ejemplos de preguntas y respuestas para demostrar precisión.

- Utilicen prompts o cadenas conversacionales simples, pero bien
  contextualizadas.

- Mantengan la presentación breve y basada en evidencia: *mostrar, no
  contar.*

**CRITERIOS DE ÉXITO**

  -----------------------------------------------------------------------
  **Criterio**           **Qué se evalúa**
  ---------------------- ------------------------------------------------
  **Precisión de las     Respuestas coherentes, sin alucinaciones,
  respuestas**           fundamentadas en el documento cargado.

  **Carga e indexación   Que el sistema permita subir y procesar
  de documentos**        múltiples archivos sin errores.

  **Interfaz y           Interacción fluida, interfaz limpia, comprensión
  usabilidad**           por parte de usuarios no técnicos.

  **Despliegue           Prototipo accesible desde URL pública, sin
  funcional**            necesidad de instalación.

  **Presentación y       Explicación concisa del flujo, demo funcional y
  claridad del pitch**   valor del proyecto.
  -----------------------------------------------------------------------

**ENTREGABLES GENERALES**

**Dataset o conjunto de documentos**

- Archivos legales usados (PDF o TXT), claramente identificados.

- Breve descripción del contenido y su propósito dentro de la demo.

**Backend y motor de IA**

- Código fuente del módulo de procesamiento (indexación, embeddings y
  retrieval).

- Archivo de configuración o notebook demostrativo.

- Instrucciones para replicar la ejecución.

**Frontend y chat funcional**

- Interfaz que permita interactuar con el asistente.

- Mostrar preguntas, respuestas y referencias textuales.

**Reporte técnico**

- Explicación breve del enfoque, arquitectura, librerías y pruebas
  realizadas.

- Al menos dos ejemplos de consultas exitosas.

**Repositorio público**

- Estructura limpia y documentación clara (README.md).

- Enlace de despliegue (URL pública funcional).

- Cargar todo el proyecto y documentación en el campus.

**PITCH FINAL**

Durante el cierre del hackathon, cada equipo dispondrá de 5 minutos para
presentar su asistente.\
La presentación deberá incluir:

1.  Breve introducción al problema.

2.  Arquitectura técnica y flujo del sistema.

3.  Demo en vivo (carga de documento y consulta real).

4.  Métricas o ejemplos de precisión.

5.  Cierre con el valor potencial de la solución.
