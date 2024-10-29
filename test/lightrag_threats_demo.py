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

# Get table structure
cursor.execute("PRAGMA table_info(parsed_content)")
columns = cursor.fetchall()
print("Table structure:", columns)

# Read and insert all content
cursor.execute('SELECT content FROM parsed_content')
contents = cursor.fetchall()
print(f"Found {len(contents)} rows in the database")

processed_count = 0
for content in contents:
    if content and isinstance(content[0], str) and content[0].strip():
        try:
            rag.insert(content[0])
            processed_count += 1
            logging.info(f"Successfully processed content chunk {processed_count}: {content[0][:50]}...")
        except Exception as e:
            logging.error(f"Error processing content: {str(e)}")
    else:
        logging.warning("Skipped empty or invalid content")

conn.close()

print(f"\nSuccessfully processed {processed_count} content chunks")

# Only proceed with queries if we have embedded content
if processed_count > 0:
    test_query = "What are the main security threats discussed in these documents?"
    
    print("\nNaive Search Results:")
    print(rag.query(test_query, param=QueryParam(mode="naive")))

    print("\nLocal Search Results:")
    print(rag.query(test_query, param=QueryParam(mode="local")))

    print("\nGlobal Search Results:")
    print(rag.query(test_query, param=QueryParam(mode="global")))

    print("\nHybrid Search Results:")
    print(rag.query(test_query, param=QueryParam(mode="hybrid")))
else:
    print("\nNo content was successfully embedded. Please check your database content.")
