import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app.core.db_config import get_connection

conn = get_connection()
cur = conn.cursor()
try:
    print("Migrating food & dining/groceries ...")
    cur.execute("UPDATE transactions SET category='Groceries', subcategory='' WHERE category='Food & Dining' AND subcategory='Groceries'")
    
    print("Migrating matching ILIKE narrations ...")
    cur.execute("""
        UPDATE transactions SET category='Groceries', subcategory='' 
        WHERE narration ILIKE '%zepto%' OR narration ILIKE '%blinkit%' 
        OR narration ILIKE '%bigbasket%' OR narration ILIKE '%instamart%' 
        OR narration ILIKE '%grocery%' OR narration ILIKE '%grofers%' 
        OR narration ILIKE '%reliance fresh%' OR narration ILIKE '%dmart%' 
        OR narration ILIKE '%jiomart%' OR narration ILIKE '%supermarket%'
    """)
    conn.commit()
    print("Database retroactive migration done")
except Exception as e:
    print("Error:", e)
cur.close()
conn.close()
