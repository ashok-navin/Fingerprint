import json
import sqlite3
import datetime
from bson import ObjectId
from pymongo import MongoClient, errors
from config import MONGODB_URI, DB_NAME, COLLECTION_NAME, BASE_DIR

class BiometricDatabase:
    def __init__(self):
        self.use_atlas = False
        self.client = None
        self.db = None
        self.collection = None
        self.status_message = ""
        self.sqlite_path = BASE_DIR / "local_biometric.db"
        
        self._init_sqlite()
        self._init_mongodb()

    def _init_sqlite(self):
        """Initialize local SQLite database for fallback and local sync."""
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    mobile TEXT NOT NULL,
                    blood_group TEXT NOT NULL,
                    emergency_contact TEXT NOT NULL,
                    address TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    fingerprint_path TEXT,
                    embeddings_json TEXT,
                    created_at TEXT
                )
            ''')
            conn.commit()

    def _init_mongodb(self):
        """Attempt connection to MongoDB Atlas."""
        try:
            # Configure DNS fallback for MongoDB SRV resolution
            try:
                import dns.resolver
                _orig_init = getattr(dns.resolver.Resolver, '_orig_init_custom', None)
                if _orig_init is None:
                    dns.resolver.Resolver._orig_init_custom = dns.resolver.Resolver.__init__
                    def _custom_resolver_init(self_res, *a, **kw):
                        dns.resolver.Resolver._orig_init_custom(self_res, *a, **kw)
                        self_res.nameservers = ['8.8.8.8', '1.1.1.1', '8.8.4.4']
                    dns.resolver.Resolver.__init__ = _custom_resolver_init
            except Exception:
                pass

            # Check if URI contains placeholders
            if "<username>" in MONGODB_URI or "<password>" in MONGODB_URI:
                self.use_atlas = False
                self.status_message = "MongoDB Atlas URI has template placeholders. Using local resilient storage (Update .env with your Atlas credentials)."
                return

            self.client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            # Trigger server connection check
            self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            self.use_atlas = True
            self.status_message = f"Connected successfully to MongoDB Atlas database '{DB_NAME}'"
            print(f"[Database] {self.status_message}")
        except Exception as e:
            self.use_atlas = False
            self.status_message = f"MongoDB Atlas offline or unreachable ({str(e)}). Running in resilient local storage mode."
            print(f"[Database] {self.status_message}")

    def get_status(self):
        """Returns connection status and user counts."""
        user_count = self.get_user_count()
        return {
            "atlas_connected": self.use_atlas,
            "database_type": "MongoDB Atlas" if self.use_atlas else "Local Storage (Resilient Fallback)",
            "status_message": self.status_message,
            "total_users": user_count,
            "db_name": DB_NAME if self.use_atlas else "local_biometric.db"
        }

    def insert_user(self, user_data):
        """
        Insert user record.
        user_data dict should contain:
        name, mobile, blood_group, emergency_contact, address, age, gender,
        fingerprint_path, embeddings (list of floats)
        """
        now = datetime.datetime.utcnow().isoformat()
        user_data['created_at'] = now
        
        # 1. Save to Atlas if available
        if self.use_atlas and self.collection is not None:
            try:
                doc = dict(user_data)
                result = self.collection.insert_one(doc)
                user_data['id'] = str(result.inserted_id)
            except Exception as e:
                print(f"[Database] MongoDB insert error: {e}, saving locally.")
                user_data['id'] = user_data.get('id') or f"usr_{int(datetime.datetime.utcnow().timestamp()*1000)}"
        else:
            user_data['id'] = user_data.get('id') or f"usr_{int(datetime.datetime.utcnow().timestamp()*1000)}"

        # 2. Always maintain local copy for high resilience & offline querying
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            embeddings_str = json.dumps(user_data.get('embeddings', []))
            
            # If multiple fingerprint paths exist, store primary in fingerprint_path and paths list in JSON
            fp_primary = user_data.get('fingerprint_path', '')
            if not fp_primary and user_data.get('fingerprint_paths'):
                fp_primary = user_data['fingerprint_paths'][0]
            
            # Store paths JSON inside embeddings or metadata if needed
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (id, name, mobile, blood_group, emergency_contact, address, age, gender, fingerprint_path, embeddings_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                str(user_data['id']),
                user_data['name'],
                user_data['mobile'],
                user_data['blood_group'],
                user_data['emergency_contact'],
                user_data['address'],
                int(user_data['age']),
                user_data['gender'],
                fp_primary,
                embeddings_str,
                user_data['created_at']
            ))
            conn.commit()

        return user_data

    def get_all_users(self):
        """Fetch all registered users with their embeddings."""
        users = []
        if self.use_atlas and self.collection is not None:
            try:
                docs = list(self.collection.find())
                for doc in docs:
                    doc['id'] = str(doc.get('_id', doc.get('id', '')))
                    if '_id' in doc:
                        del doc['_id']
                    if 'fingerprint_paths' not in doc and 'fingerprint_path' in doc:
                        doc['fingerprint_paths'] = [doc['fingerprint_path']] if doc['fingerprint_path'] else []
                    users.append(doc)
                return users
            except Exception as e:
                print(f"[Database] Failed to read from Atlas ({e}), reading local SQLite.")

        # Read from local SQLite
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users ORDER BY created_at DESC')
            rows = cursor.fetchall()
            for r in rows:
                embeddings = []
                if r['embeddings_json']:
                    try:
                        embeddings = json.loads(r['embeddings_json'])
                    except Exception:
                        embeddings = []
                fp_path = r['fingerprint_path'] or ''
                users.append({
                    "id": r['id'],
                    "name": r['name'],
                    "mobile": r['mobile'],
                    "blood_group": r['blood_group'],
                    "emergency_contact": r['emergency_contact'],
                    "address": r['address'],
                    "age": r['age'],
                    "gender": r['gender'],
                    "fingerprint_path": fp_path,
                    "fingerprint_paths": [fp_path] if fp_path else [],
                    "embeddings": embeddings,
                    "created_at": r['created_at']
                })
        return users

    def get_user_by_id(self, user_id):
        """Fetch a single user profile."""
        if self.use_atlas and self.collection is not None:
            try:
                query = {"_id": ObjectId(user_id)} if ObjectId.is_valid(user_id) else {"id": user_id}
                doc = self.collection.find_one(query)
                if doc:
                    doc['id'] = str(doc.get('_id', doc.get('id', '')))
                    if '_id' in doc:
                        del doc['_id']
                    return doc
            except Exception:
                pass

        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            r = cursor.fetchone()
            if r:
                embeddings = []
                if r['embeddings_json']:
                    try:
                        embeddings = json.loads(r['embeddings_json'])
                    except Exception:
                        embeddings = []
                return {
                    "id": r['id'],
                    "name": r['name'],
                    "mobile": r['mobile'],
                    "blood_group": r['blood_group'],
                    "emergency_contact": r['emergency_contact'],
                    "address": r['address'],
                    "age": r['age'],
                    "gender": r['gender'],
                    "fingerprint_path": r['fingerprint_path'],
                    "embeddings": embeddings,
                    "created_at": r['created_at']
                }
        return None

    def delete_user(self, user_id):
        """Delete a user record."""
        deleted = False
        if self.use_atlas and self.collection is not None:
            try:
                query = {"_id": ObjectId(user_id)} if ObjectId.is_valid(user_id) else {"id": user_id}
                self.collection.delete_one(query)
                deleted = True
            except Exception as e:
                print(f"[Database] Delete from Atlas error: {e}")

        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            conn.commit()
            if cursor.rowcount > 0:
                deleted = True
        return deleted

    def update_user_embeddings(self, user_id, embeddings_list):
        """Update embeddings for an existing user record in both Atlas and SQLite."""
        if self.use_atlas and self.collection is not None:
            try:
                query = {"_id": ObjectId(user_id)} if ObjectId.is_valid(user_id) else {"id": user_id}
                self.collection.update_one(query, {"$set": {"embeddings": embeddings_list}})
            except Exception as e:
                print(f"[Database] Error updating Atlas embeddings: {e}")

        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            embeddings_str = json.dumps(embeddings_list)
            cursor.execute('UPDATE users SET embeddings_json = ? WHERE id = ?', (embeddings_str, str(user_id)))
            conn.commit()

    def get_user_count(self):
        """Get total user count."""
        if self.use_atlas and self.collection is not None:
            try:
                return self.collection.count_documents({})
            except Exception:
                pass
        with sqlite3.connect(self.sqlite_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users')
            count = cursor.fetchone()[0]
            return count

# Global database instance
db = BiometricDatabase()

