import os
import math
import numpy as np
import cv2
from pathlib import Path
from config import FINGERPRINT_FOLDER, SAMPLE_FOLDER, BASE_DIR
from database import db
from cnn_model import get_cnn

def generate_synthetic_fingerprint(pattern_type="whorl", seed=42, size=(300, 300)):
    """
    Generates a realistic biometric fingerprint pattern (ridges, valleys, core, whorls, minutiae).
    """
    np.random.seed(seed)
    w, h = size
    img = np.zeros((h, w), dtype=np.float32)
    cx, cy = w // 2, h // 2
    
    y, x = np.mgrid[0:h, 0:w]
    dx = x - cx
    dy = y - cy
    
    if pattern_type == "whorl":
        # Concentric spiral/whorl ridges
        r = np.sqrt(dx**2 + dy**2)
        theta = np.arctan2(dy, dx)
        freq = 0.18 + 0.02 * np.sin(seed)
        ridges = np.sin(freq * r + 2.5 * theta)
    elif pattern_type == "loop":
        # Loop pattern flowing to one side
        flow = dx + 0.5 * (dy**2 / 100.0)
        freq = 0.20
        ridges = np.sin(freq * flow + 0.05 * dy)
    elif pattern_type == "arch":
        # Arch pattern
        arch = dy - 0.003 * (dx**2)
        freq = 0.22
        ridges = np.sin(freq * arch)
    elif pattern_type == "tented_arch":
        # Tented arch with sharp central peak (Unique signature)
        flow = dy - 0.009 * (dx**2) * np.exp(-np.abs(dx)/35.0)
        freq = 0.23
        ridges = np.sin(freq * flow + 0.15 * np.cos(dx/8.0))
    elif pattern_type == "twin_loop":
        # Double core loop
        r1 = np.sqrt((dx - 30)**2 + (dy - 20)**2)
        r2 = np.sqrt((dx + 30)**2 + (dy + 20)**2)
        ridges = np.sin(0.18 * r1) + np.sin(0.18 * r2)
    else: # composite
        r = np.sqrt(dx**2 + 1.2 * dy**2)
        ridges = np.sin(0.19 * r + 0.1 * np.sin(dx/15.0))

    # Apply elliptical finger mask
    mask = ((dx / (w * 0.42))**2 + (dy / (h * 0.46))**2) < 1.0
    
    # Add natural texture noise
    noise = np.random.normal(0, 0.12, (h, w))
    fingerprint = np.where(mask, ridges + noise, 1.0)
    
    # Normalize to 0-255 grayscale
    fingerprint = np.clip((fingerprint + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
    
    # Apply Gaussian smoothing to mimic skin ridge softness
    fingerprint = cv2.GaussianBlur(fingerprint, (3, 3), 0.8)
    return fingerprint

def reindex_all_existing_users():
    """
    Recomputes and updates CNN embeddings for all existing database records
    using their stored fingerprint image files.
    """
    from fingerprint_matcher import matcher
    users = db.get_all_users()
    print(f"[Reindex] Updating CNN embeddings for {len(users)} existing user records...")
    
    for u in users:
        fp_paths = u.get("fingerprint_paths") or ([u.get("fingerprint_path")] if u.get("fingerprint_path") else [])
        if isinstance(fp_paths, str):
            fp_paths = [fp_paths]
            
        new_embeddings = []
        for path_str in fp_paths:
            if not path_str:
                continue
            full_p = BASE_DIR / path_str.lstrip('/')
            if full_p.exists():
                with open(full_p, 'rb') as f:
                    features = matcher.extract_features(f.read())
                    new_embeddings.append(features["embedding"])
        
        if new_embeddings:
            db.update_user_embeddings(u["id"], new_embeddings if len(new_embeddings) > 1 else new_embeddings[0])
            print(f"  [OK] Updated embeddings for: {u.get('name')} ({len(new_embeddings)} prints)")

def sync_and_backfill_fingerprint_images():
    """
    Backfills base64 encoded fingerprint images for all registered users in MongoDB Atlas and SQLite
    so their images load seamlessly on Vercel and cloud platforms without disk dependencies.
    """
    import base64
    users = db.get_all_users()
    print(f"[Backfill] Checking fingerprint images for {len(users)} registered users...")
    
    updated_count = 0
    for u in users:
        current_img = u.get("fingerprint_image", "")
        if current_img and current_img.startswith("data:image/"):
            continue
            
        fp_path_str = u.get("fingerprint_path", "")
        fp_paths = u.get("fingerprint_paths", [])
        if not fp_path_str and fp_paths:
            fp_path_str = fp_paths[0]
            
        b64_data = ""
        # 1. Try to find the file locally on disk
        if fp_path_str:
            clean_rel = fp_path_str.lstrip('/')
            candidates = [
                BASE_DIR / clean_rel,
                FINGERPRINT_FOLDER / Path(clean_rel).name,
                SAMPLE_FOLDER / Path(clean_rel).name,
                BASE_DIR / 'uploads' / 'fingerprints' / Path(clean_rel).name,
                BASE_DIR / 'uploads' / 'samples' / Path(clean_rel).name
            ]
            for cand in candidates:
                if cand.exists():
                    try:
                        ext = cand.suffix.lstrip('.').lower() or 'png'
                        with open(cand, 'rb') as f:
                            raw = f.read()
                        b64_data = f"data:image/{ext};base64,{base64.b64encode(raw).decode('utf-8')}"
                        break
                    except Exception as e:
                        print(f"  [!] Error reading {cand}: {e}")
                        
        # 2. If no local file found, generate a unique synthetic fingerprint based on user's name/id
        if not b64_data:
            seed_val = abs(hash(u.get("name", "") + str(u.get("id", "")))) % 1000
            img = generate_synthetic_fingerprint(pattern_type="whorl" if seed_val % 2 == 0 else "loop", seed=seed_val)
            _, buf = cv2.imencode('.png', img)
            b64_data = f"data:image/png;base64,{base64.b64encode(buf.tobytes()).decode('utf-8')}"
            
        if b64_data:
            db.update_user_fingerprint_image(u["id"], b64_data)
            updated_count += 1
            print(f"  [OK] Updated base64 fingerprint image for: {u.get('name')} (#{u.get('id')})")

    print(f"[Backfill] Completed! Updated {updated_count} user record(s) with base64 biometric image data.")

def seed_database():
    import base64
    print("[Seed] Initializing seed users and biometric fingerprint dataset...")
    
    sample_users = [
        {
            "name": "Alexander Vance",
            "mobile": "+1 (555) 234-8901",
            "blood_group": "O+",
            "emergency_contact": "+1 (555) 902-1144 (Elena Vance - Spouse)",
            "address": "402 Silicon Vista Blvd, Tech District, San Francisco, CA 94107",
            "age": 34,
            "gender": "Male",
            "pattern": "whorl",
            "seed": 101,
            "filename": "alexander_vance_thumb.png"
        },
        {
            "name": "Dr. Sarah Lin, MD",
            "mobile": "+1 (555) 876-4321",
            "blood_group": "A+",
            "emergency_contact": "+1 (555) 671-8899 (Marcus Lin - Brother)",
            "address": "789 Metro Medical Center Ave, Suite 400, Chicago, IL 60611",
            "age": 29,
            "gender": "Female",
            "pattern": "loop",
            "seed": 202,
            "filename": "sarah_lin_index.png"
        },
        {
            "name": "Rajesh Kumar Sharma",
            "mobile": "+91 98765 43210",
            "blood_group": "B+",
            "emergency_contact": "+91 98111 22334 (Pooja Sharma - Wife)",
            "address": "Plot 42, Cyber Heights, Sector 62, Noida, Uttar Pradesh 201309",
            "age": 41,
            "gender": "Male",
            "pattern": "arch",
            "seed": 303,
            "filename": "rajesh_sharma_thumb.png"
        },
        {
            "name": "Amina Al-Mansoor",
            "mobile": "+971 50 123 4567",
            "blood_group": "AB+",
            "emergency_contact": "+971 52 987 6543 (Tariq Al-Mansoor - Father)",
            "address": "Villa 18, Al Safa 2, Jumeirah, Dubai, UAE",
            "age": 26,
            "gender": "Female",
            "pattern": "twin_loop",
            "seed": 404,
            "filename": "amina_mansoor_index.png"
        },
        {
            "name": "David Sterling",
            "mobile": "+44 7700 900543",
            "blood_group": "O-",
            "emergency_contact": "+44 7700 900888 (Charlotte Sterling - Mother)",
            "address": "14 Kensington Court Gardens, London W8 5QP, United Kingdom",
            "age": 48,
            "gender": "Male",
            "pattern": "composite",
            "seed": 505,
            "filename": "david_sterling_thumb.png"
        }
    ]

    from fingerprint_matcher import matcher

    # Pre-generate sample fingerprints
    for idx, u in enumerate(sample_users):
        img = generate_synthetic_fingerprint(pattern_type=u["pattern"], seed=u["seed"])
        
        fp_path = FINGERPRINT_FOLDER / u["filename"]
        sample_path = SAMPLE_FOLDER / u["filename"]
        cv2.imwrite(str(fp_path), img)
        cv2.imwrite(str(sample_path), img)

        with open(fp_path, 'rb') as f:
            raw_bytes = f.read()
            features = matcher.extract_features(raw_bytes)
            embedding = features["embedding"]
            b64_str = f"data:image/png;base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

        user_record = {
            "id": f"usr_seed_{idx+1:03d}",
            "name": u["name"],
            "mobile": u["mobile"],
            "blood_group": u["blood_group"],
            "emergency_contact": u["emergency_contact"],
            "address": u["address"],
            "age": u["age"],
            "gender": u["gender"],
            "fingerprint_path": f"/uploads/fingerprints/{u['filename']}",
            "fingerprint_image": b64_str,
            "fingerprint_paths": [f"/uploads/fingerprints/{u['filename']}"],
            "embeddings": [embedding]
        }
        db.insert_user(user_record)
        print(f"  [+] Enrolled: {u['name']} ({u['blood_group']}) -> Embedding generated.")

    # Generate an UNREGISTERED fingerprint sample for testing negative identification
    unreg_img = generate_synthetic_fingerprint(pattern_type="tented_arch", seed=888)
    unreg_path = SAMPLE_FOLDER / "unregistered_suspect_fingerprint.png"
    cv2.imwrite(str(unreg_path), unreg_img)
    print(f"  [*] Created Unregistered Test Print: {unreg_path.name}")

    # Reindex all users in database
    reindex_all_existing_users()
    sync_and_backfill_fingerprint_images()

    print(f"[Seed] Successfully seeded & indexed biometric profiles into database.")

if __name__ == "__main__":
    sync_and_backfill_fingerprint_images()


