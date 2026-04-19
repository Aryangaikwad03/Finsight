import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app.core.db_config import get_connection

conn = get_connection()
cur = conn.cursor()
try:
    cur.execute("ALTER TABLE transactions ADD COLUMN IF NOT EXISTS subcategory VARCHAR(60)")
    conn.commit()
    print("Successfully added subcategory to transactions")
except Exception as e:
    print("Error:", e)
cur.close()
conn.close()
