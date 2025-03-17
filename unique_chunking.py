import psycopg2
import uuid
import hashlib
from sentence_transformers import SentenceTransformer  # Or your preferred embedding model
import psycopg2.extras
import json

# Database connection details (REPLACE WITH YOUR CREDENTIALS)
DB_HOST = "your_db_host"
DB_NAME = "your_db_name"
DB_USER = "your_db_user"
DB_PASSWORD = "your_db_password"

# Initialize embedding model
model = SentenceTransformer('all-mpnet-base-v2')  # Example model, choose as needed

BATCH_SIZE_DOCUMENTS = 50  # Number of documents to process in each batch
BATCH_SIZE_DB = 500       # Number of database operations (inserts/updates/deletes) to batch

def connect_to_db():
    """Establishes a connection to the PostgreSQL database."""
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

def process_documents(documents):  # documents is a list of tuples like (page_id, document_text, metadata)
    """Processes a list of documents, chunking, embedding, and storing them in the database."""
    conn = connect_to_db()
    if not conn:
        return

    try:
        cur = conn.cursor()

        # Iterate through documents in batches
        for i in range(0, len(documents), BATCH_SIZE_DOCUMENTS):
            document_batch = documents[i:i + BATCH_SIZE_DOCUMENTS] # Get a batch of documents
            vectors_to_insert = [] # List to store vectors to be inserted
            vectors_to_update = [] # List to store vectors to be updated
            vectors_to_delete = [] # List to store chunk_ids of vectors to be deleted

            # Process each document in the batch
            for page_id, document_text, metadata in document_batch:
                chunks = split_into_sentences(document_text)  # Chunk the document into sentences

                # Retrieve existing chunks for the current document from the database
                cur.execute("SELECT chunk_id, chunk_index, content_hash FROM embeddings WHERE document_id = %s ORDER BY chunk_index", (page_id,))
                existing_chunks = cur.fetchall()

                # Create dictionaries for efficient lookup of existing chunks
                chunk_hash_map = {row[1]: row[2] for row in existing_chunks} # chunk_index: content_hash
                existing_chunk_ids = {row[1]: row[0] for row in existing_chunks} # chunk_index: chunk_id

                # Identify chunks that need to be deleted
                chunks_to_delete_indices = set(chunk_hash_map.keys()) - set(range(len(chunks))) # Find chunk_indexes that are in the database, but not in the new chunk list.
                for chunk_index in chunks_to_delete_indices:
                    vectors_to_delete.append(existing_chunk_ids[chunk_index]) # Add chunk_id to the delete list

                # Process each chunk in the document
                for j, chunk_text in enumerate(chunks):
                    new_hash = hashlib.sha256(chunk_text.encode()).hexdigest() # Calculate the hash of the chunk
                    embedding = model.encode(chunk_text) # Generate the embedding for the chunk

                    if j in chunk_hash_map:  # Chunk already exists (update)
                        chunk_id = existing_chunk_ids[j] # Get the chunk_id from the existing chunk data
                        stored_hash = chunk_hash_map[j] # Get the stored hash
                        if new_hash != stored_hash: # Check if the content has changed
                            vectors_to_update.append((chunk_id, embedding, json.dumps(metadata | {'content_hash': new_hash}))) # Add the update to the update list.
                    else:  # New chunk (insert)
                        chunk_id = str(uuid.uuid4()) # Generate a new UUID
                        vectors_to_insert.append((chunk_id, page_id, j, new_hash, embedding, json.dumps(metadata | {'content_hash': new_hash}))) # Add the insert to the insert list.

            # Batch operations (insert, update, delete)
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

            conn.commit() # Commit the changes for the document batch.

    except psycopg2.Error as e:
        conn.rollback() # Rollback changes if there is an error.
        print(f"Error processing documents: {e}")
    finally:
        if conn:
            cur.close() # Close the cursor
            conn.close() # Close the connection

def split_into_sentences(text): # Replace with your actual sentence splitter
    """Placeholder - Replace with your actual sentence splitting logic."""
    return text.split(". ") # Example: Splitting by ". "

# Example usage (REPLACE WITH YOUR DATA)
documents = [
    ("doc1", "This is the first sentence. This is the second.", {"title": "Doc 1", "author": "John Doe"}),
    ("doc2", "Sentence one here. Sentence two.", {"title": "Doc 2", "author": "Jane Smith"}),
    # ... more documents
]

process_documents(documents)
