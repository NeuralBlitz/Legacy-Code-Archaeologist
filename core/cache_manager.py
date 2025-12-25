import sqlite3
import hashlib

class CacheManager:
    def __init__(self, db_path="archeology_cache.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_cache 
            (file_hash TEXT PRIMARY KEY, ai_summary TEXT, complexity_score INTEGER, tags TEXT)
        """)

    def get(self, content):
        h = hashlib.md5(content.encode()).hexdigest()
        row = self.conn.execute("SELECT ai_summary, complexity_score, tags FROM file_cache WHERE file_hash=?", (h,)).fetchone()
        if row:
            return {"summary": row[0], "complexity_score": row[1], "tags": row[2].split(",")}
        return None

    def save(self, content, data):
        h = hashlib.md5(content.encode()).hexdigest()
        self.conn.execute("INSERT OR REPLACE INTO file_cache VALUES (?,?,?,?)", 
            (h, data['summary'], data['complexity_score'], ",".join(data['tags'])))
        self.conn.commit()
