"""
debug_mongo.py
Run this to see exactly what's in your MongoDB.
python debug_mongo.py
"""
from pymongo import MongoClient

MONGO_URI = "mongodb+srv://jadhavsushant379_db_user:EjRiiekC4N1iZHg5@cluster0.f4zpb4k.mongodb.net/"

print("Connecting to MongoDB...")
client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)

print("\n=== ALL DATABASES ===")
for db_name in client.list_database_names():
    print(f"  DB: {db_name}")
    db = client[db_name]
    for col_name in db.list_collection_names():
        count = db[col_name].count_documents({})
        print(f"      Collection: '{col_name}' → {count} documents")

print("\n=== SAMPLE DOCUMENT (from recruitment_db.candidates) ===")
try:
    col = client["recruitment_db"]["candidates"]
    doc = col.find_one({})
    if doc:
        print(f"Found! Keys: {list(doc.keys())}")
        # Show structure
        if "output" in doc:
            print(f"  output keys: {list(doc['output'].keys())}")
    else:
        print("Collection exists but is EMPTY")
except Exception as e:
    print(f"Error: {e}")
