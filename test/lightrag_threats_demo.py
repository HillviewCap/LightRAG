import os
import logging
import sqlite3
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./lightrag_cache"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Initialize LightRAG with Ollama
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="gemma:2b",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

# Connect to SQLite database
conn = sqlite3.connect('threats.db')
cursor = conn.cursor()

# Read and insert all threat descriptions
cursor.execute('SELECT description FROM threats')
threats = cursor.fetchall()
for threat in threats:
    rag.insert(threat[0])

conn.close()

# Example queries using different search modes
test_query = "What are common network-based attacks?"

print("\nNaive Search Results:")
print(rag.query(test_query, param=QueryParam(mode="naive")))

print("\nLocal Search Results:")
print(rag.query(test_query, param=QueryParam(mode="local")))

print("\nGlobal Search Results:")
print(rag.query(test_query, param=QueryParam(mode="global")))

print("\nHybrid Search Results:")
print(rag.query(test_query, param=QueryParam(mode="hybrid")))
