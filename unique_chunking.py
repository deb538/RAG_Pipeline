import psycopg2
import uuid
import hashlib
from sentence_transformers import SentenceTransformer
import psycopg2.extras
import json

# Database connection details (REPLACE WITH YOUR CREDENTIALS)
DB_HOST = "your_db_host"
DB_NAME = "your_db_name"
DB_USER = "your_db_user"
DB_PASSWORD = "your_db_password"

# Initialize embedding model
model = SentenceTransformer('all-mpnet-base-v2')

BATCH_SIZE_DOCUMENTS = 50
BATCH_SIZE_DB = 500

def connect_to_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def process_documents(documents):
    conn = connect_to_db()
    if not conn:
        return

    try:
        cur = conn.cursor()

        for page_id, document_text, metadata in documents:
            chunks = split_into_sentences(document_text)

            cur.execute("SELECT chunk_id, chunk_index, content_hash FROM embeddings WHERE document_id = %s ORDER BY chunk_index", (page_id,))
            existing_chunks = cur.fetchall()

            vectors_to_insert = []
            vectors_to_update = []
            vectors_to_delete = []

            # 1. Iterate through existing chunks
            for existing_chunk_id, existing_chunk_index, existing_chunk_hash in existing_chunks:
                # 2. Compare content hashes
                if existing_chunk_index < len(chunks):
                    new_chunk_hash = hashlib.sha256(chunks[existing_chunk_index].encode()).hexdigest()
                    if new_chunk_hash != existing_chunk_hash:
                        # Chunk has been modified, update it
                        vectors_to_update.append((existing_chunk_id, model.encode(chunks[existing_chunk_index]), json.dumps(metadata | {'content_hash': new_chunk_hash})))
                else:
                    # No new chunk with this index, delete the existing chunk
                    vectors_to_delete.append(existing_chunk_id)

            # process new chunks.
            for j, chunk_text in enumerate(chunks):
                if j >= len(existing_chunks):
                    new_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
                    vectors_to_insert.append((str(uuid.uuid4()), page_id, j, new_hash, model.encode(chunk_text), json.dumps(metadata | {'content_hash': new_hash})))

            # Batch operations
            if vectors_to_insert:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    INSERT INTO embeddings (chunk_id, document_id, chunk_index, content_hash, embedding, metadata)
                    VALUES %s::vector
                    """,
                    vectors_to_insert,
                    template="""(%(0)s, %(1)s, %(2)s, %(3)s, %(4)s, %(5)s::jsonb)""",
                    page_size=BATCH_SIZE_DB
                )

            if vectors_to_update:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    UPDATE embeddings
                    SET embedding = %s::vector, metadata = %s
                    WHERE chunk_id = %s
                    """,
                    [(embedding, metadata, chunk_id) for chunk_id, embedding, metadata in vectors_to_update],
                    template="""(%(0)s, %(1)s, %(2)s)""",
                    page_size=BATCH_SIZE_DB
                )

            if vectors_to_delete:
                psycopg2.extras.execute_values(
                    cur,
                    """
                    DELETE FROM embeddings WHERE chunk_id = %s
                    """,
                    [(chunk_id,) for chunk_id in vectors_to_delete],
                    template="""(%(0)s)""",
                    page_size=BATCH_SIZE_DB
                )

            conn.commit()

    except psycopg2.Error as e:
        conn.rollback()
        print(f"Error processing documents: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

def split_into_sentences(text):
    # Replace with your actual sentence splitting logic
    return text.split(". ")

# Example usage (REPLACE WITH YOUR DATA)
documents = [
    ("doc1", "This is the first sentence. This is the second.", {"title": "Doc 1", "author": "John Doe"}),
    ("doc2", "Sentence one here. Sentence two. and a third one", {"title": "Doc 2", "author": "Jane Smith"}),
    ("doc3", "only one sentence", {"title": "Doc3", "author": "Anonymous"})
]

process_documents(documents)
