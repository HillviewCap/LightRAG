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
    llm_model_name="gemma2:2b",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://10.0.10.9:11434/", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://10.0.10.9:11434/"
        ),
    ),
)

# Connect to SQLite database
conn = sqlite3.connect('threats.db')
cursor = conn.cursor()

# Inspect database tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Available tables in the database:", tables)

# Read and insert all content
cursor.execute('SELECT content FROM parsed_content')
contents = cursor.fetchall()
for content in contents:
    if content[0]:  # Check if content is not None/empty
        rag.insert(content[0])
        logging.info("Processed and inserted content chunk")

conn.close()

# Example queries using different search modes
test_query = "What are the main security threats discussed in these documents?"

print("\nNaive Search Results:")
print(rag.query(test_query, param=QueryParam(mode="naive")))

print("\nLocal Search Results:")
print(rag.query(test_query, param=QueryParam(mode="local")))

print("\nGlobal Search Results:")
print(rag.query(test_query, param=QueryParam(mode="global")))

print("\nHybrid Search Results:")
print(rag.query(test_query, param=QueryParam(mode="hybrid")))
