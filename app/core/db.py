import sqlite3
import os
import logging

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "feedback.db")

def init_db():
    """Initialize the SQLite database with the required tables."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                model_response TEXT NOT NULL,
                is_liked BOOLEAN NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        logger.info(f"Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

def insert_feedback(prompt: str, model_response: str, is_liked: bool):
    """Insert a new feedback record."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (prompt, model_response, is_liked, status)
            VALUES (?, ?, ?, ?)
        ''', (prompt, model_response, is_liked, 'pending'))
        
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to insert feedback: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def get_pending_feedback():
    """Get all pending feedback records."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, prompt, model_response, is_liked 
            FROM feedback 
            WHERE status = 'pending'
        ''')
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to get pending feedback: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def mark_feedback_processed(ids: list):
    """Mark specific feedback records as processed."""
    if not ids:
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' for _ in ids)
        cursor.execute(f'''
            UPDATE feedback 
            SET status = 'processed' 
            WHERE id IN ({placeholders})
        ''', ids)
        
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to mark feedback as processed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
