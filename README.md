# Building a RAG application from scratch

Esta es una guía paso a paso para crear una aplicación RAG (Recuperación-Generación Aumentada) sencilla con Pinecone y la API de OpenAI. La aplicación te permitirá hacer preguntas sobre cualquier vídeo de YouTube.

## Setup

1. Crea un entorno virtual e instale los paquetes necesarios:

```bash
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

2. Crea una cuenta gratuita de Pinecone y obtén tu clave API aquí (https://www.pinecone.io/).

3. Crea un archivo `.env` con las siguientes variables:

```bash
OPENAI_API_KEY = 
PINECONE_API_KEY = 
PINECONE_API_ENV = 
```