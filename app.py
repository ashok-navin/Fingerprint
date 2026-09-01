import os
import uuid
import json
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, Response
from werkzeug.utils import secure_filename

from config import (
    PORT, DEBUG, SECRET_KEY, UPLOAD_FOLDER, 
    FINGERPRINT_FOLDER, SAMPLE_FOLDER, MATCH_THRESHOLD, BASE_DIR,
    ENROLL_PASSWORD
)
from database import db
from fingerprint_matcher import matcher

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    # 1. If file exists on local filesystem, serve it
    file_path = UPLOAD_FOLDER / filename
    if file_path.exists():
        return send_from_directory(str(UPLOAD_FOLDER), filename)
    
    # 2. Check if this filename is stored in database as base64
    try:
        users = db.get_all_users()
        for u in users:
            fp_p = u.get('fingerprint_path', '')
            fp_list = u.get('fingerprint_paths', [])
            fp_img = u.get('fingerprint_image', '')
            if filename in fp_p or any(filename in p for p in fp_list):
                if fp_img and fp_img.startswith('data:image/'):
                    header, data = fp_img.split(',', 1)
                    mime = header.split(';')[0].split(':')[1] if ':' in header else 'image/png'
                    raw_bytes = base64.b64decode(data)
                    return Response(raw_bytes, mimetype=mime)
                break
    except Exception as e:
        print(f"[Uploads] Lookup error: {e}")

    # 3. Dynamic Biometric Synthesizer Fallback for Vercel/Cloud
    try:
        from seed_data import generate_synthetic_fingerprint
        import cv2
        img = generate_synthetic_fingerprint(seed=abs(hash(filename)) % 1000)
        _, buf = cv2.imencode('.png', img)
        return Response(buf.tobytes(), mimetype='image/png')
    except Exception as e:
        print(f"[Uploads] Fallback generator error: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route('/api/verify-passcode', methods=['POST'])
def verify_passcode():
    """Verifies security passcode for accessing protected tabs (Directory, Database Config, Enrollment)."""
    try:
        data = request.get_json() or {}
        passcode = data.get('passcode', '').strip()
        if passcode == ENROLL_PASSWORD:
            return jsonify({"success": True, "authorized": True, "message": "Access Granted"})
        return jsonify({
            "success": False, 
            "authorized": False, 
            "error": "Authentication Failed: Incorrect security passcode. Access Denied."
        }), 403
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/db-status', methods=['GET'])
def get_db_status():
    status = db.get_status()
    return jsonify(status)

@app.route('/api/users', methods=['GET'])
def list_users():
    users = db.get_all_users()
    # Strip raw embedding vectors from public listing to keep payload light
    sanitized = []
    for u in users:
        item = dict(u)
        item.pop('embeddings', None)
        sanitized.append(item)
    return jsonify({
        "success": True,
        "count": len(sanitized),
        "users": sanitized
    })

@app.route('/api/users/<user_id>', methods=['GET'])
def get_user(user_id):
    user = db.get_user_by_id(user_id)
    if not user:
        return jsonify({"success": False, "error": "User not found"}), 404
    user_copy = dict(user)
    user_copy.pop('embeddings', None)
    return jsonify({"success": True, "user": user_copy})

@app.route('/api/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    success = db.delete_user(user_id)
    if success:
        return jsonify({"success": True, "message": "User deleted successfully"})
    return jsonify({"success": False, "error": "User not found or deletion failed"}), 404

@app.route('/api/samples', methods=['GET'])
def list_samples():
    """Returns available sample fingerprint images for 1-click test scanning."""
    samples = []
    if SAMPLE_FOLDER.exists():
        for file in SAMPLE_FOLDER.glob('*.*'):
            if file.suffix.lower().replace('.', '') in ALLOWED_EXTENSIONS:
                label = file.stem.replace('_', ' ').title()
                is_unreg = "unregistered" in file.stem.lower()
                samples.append({
                    "filename": file.name,
                    "url": f"/uploads/samples/{file.name}",
                    "label": label,
                    "is_unregistered": is_unreg
                })
    return jsonify({"success": True, "samples": samples})

@app.route('/api/identify', methods=['POST'])
def identify_fingerprint():
    """
    Accepts:
      1. Multipart file upload: 'fingerprint_image'
      2. JSON payload: {'image_data': 'data:image/png;base64,...'} or {'sample_filename': 'alexander_vance_thumb.png'}
    """
    image_bytes = None

    # Option A: File upload
    if 'fingerprint_image' in request.files:
        file = request.files['fingerprint_image']
        if file and file.filename != '' and allowed_file(file.filename):
            image_bytes = file.read()
        else:
            return jsonify({"success": False, "error": "Invalid or missing fingerprint image file."}), 400

    # Option B: JSON payload (base64 or sample filename)
    elif request.is_json:
        data = request.get_json()
        if 'sample_filename' in data:
            sample_path = SAMPLE_FOLDER / secure_filename(data['sample_filename'])
            if sample_path.exists():
                with open(sample_path, 'rb') as f:
                    image_bytes = f.read()
            else:
                return jsonify({"success": False, "error": "Sample file not found."}), 404
        elif 'image_data' in data:
            raw_b64 = data['image_data']
            if ',' in raw_b64:
                raw_b64 = raw_b64.split(',', 1)[1]
            try:
                image_bytes = base64.b64decode(raw_b64)
            except Exception:
                return jsonify({"success": False, "error": "Invalid base64 image data."}), 400

    if not image_bytes:
        return jsonify({"success": False, "error": "No fingerprint image provided."}), 400

    try:
        result = matcher.identify(image_bytes)
        result["success"] = True
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Scanning error: {str(e)}"}), 500

@app.route('/api/enroll', methods=['POST'])
def enroll_user():
    """
    Enrolls a new user with full details & multiple fingerprint scans (up to 10 images per user).
    Requires security password authorization (psr123).
    """
    try:
        # Check security passcode authorization
        auth_password = request.form.get('password', '').strip()
        if auth_password != ENROLL_PASSWORD:
            return jsonify({
                "success": False,
                "error": "Authentication Failed: Incorrect security passcode. Authorization required to add a new person."
            }), 403

        name = request.form.get('name', '').strip()
        mobile = request.form.get('mobile', '').strip()
        blood_group = request.form.get('blood_group', '').strip()
        emergency_contact = request.form.get('emergency_contact', '').strip()
        address = request.form.get('address', '').strip()
        age = request.form.get('age', '').strip()
        gender = request.form.get('gender', '').strip()

        # Validation
        if not all([name, mobile, blood_group, emergency_contact, address, age, gender]):
            return jsonify({
                "success": False, 
                "error": "All fields are required (Name, Mobile, Blood Group, Emergency Contact, Address, Age, Gender)."
            }), 400

        try:
            age_int = int(age)
            if age_int < 1 or age_int > 130:
                raise ValueError()
        except ValueError:
            return jsonify({"success": False, "error": "Age must be a valid number between 1 and 130."}), 400

        # Collect uploaded files (supports single or multi-file upload)
        uploaded_files = []
        if 'fingerprint_images' in request.files:
            uploaded_files = request.files.getlist('fingerprint_images')
        elif 'fingerprint_image' in request.files:
            uploaded_files = request.files.getlist('fingerprint_image')

        valid_files = [f for f in uploaded_files if f and f.filename != '' and allowed_file(f.filename)]
        if not valid_files:
            return jsonify({"success": False, "error": "Please provide at least one valid fingerprint image file (PNG, JPG, BMP, TIF)."}), 400

        embeddings_list = []
        fingerprint_paths = []
        total_minutiae = 0
        primary_b64 = ""

        # Process each uploaded fingerprint image (up to 10 images)
        for idx, file in enumerate(valid_files[:10]):
            image_bytes = file.read()
            features = matcher.extract_features(image_bytes)
            embeddings_list.append(features["embedding"])
            total_minutiae += features["minutiae_count"]

            ext = file.filename.rsplit('.', 1)[1].lower()
            if idx == 0:
                primary_b64 = f"data:image/{ext};base64,{base64.b64encode(image_bytes).decode('utf-8')}"

            unique_name = f"{secure_filename(name).lower()}_scan{idx+1}_{uuid.uuid4().hex[:6]}.{ext}"
            try:
                save_path = FINGERPRINT_FOLDER / unique_name
                with open(save_path, 'wb') as f:
                    f.write(image_bytes)

                sample_copy = SAMPLE_FOLDER / unique_name
                with open(sample_copy, 'wb') as f:
                    f.write(image_bytes)
            except Exception:
                pass

            fingerprint_paths.append(f"/uploads/fingerprints/{unique_name}")

        user_record = {
            "id": f"usr_{uuid.uuid4().hex[:10]}",
            "name": name,
            "mobile": mobile,
            "blood_group": blood_group.upper(),
            "emergency_contact": emergency_contact,
            "address": address,
            "age": age_int,
            "gender": gender,
            "fingerprint_path": fingerprint_paths[0] if fingerprint_paths else "",
            "fingerprint_image": primary_b64,
            "fingerprint_paths": fingerprint_paths,
            "embeddings": embeddings_list
        }

        saved = db.insert_user(user_record)
        
        response_user = dict(saved)
        response_user.pop('embeddings', None)

        return jsonify({
            "success": True,
            "message": f"Successfully enrolled {name} with {len(fingerprint_paths)} fingerprint scan(s) into Biometric Database.",
            "user": response_user,
            "total_scans_enrolled": len(fingerprint_paths),
            "avg_minutiae_count": round(total_minutiae / len(valid_files), 1)
        })

    except Exception as e:
        return jsonify({"success": False, "error": f"Enrollment error: {str(e)}"}), 500

@app.route('/api/import-dataset', methods=['POST'])
def import_dataset():
    """
    Scans the dataset/ directory for subfolders (one per person) or standard biometric dataset format,
    extracts multi-fingerprint embeddings, and enrolls each person into MongoDB Atlas.
    """
    from config import DATASET_FOLDER
    if not DATASET_FOLDER.exists():
        return jsonify({"success": False, "error": "dataset/ directory not found."}), 404

    imported_count = 0
    errors_list = []

    # 1. Check person subfolders (e.g. dataset/John_Doe/1.bmp ... 10.bmp)
    person_dirs = [d for d in DATASET_FOLDER.iterdir() if d.is_dir()]
    for pdir in person_dirs:
        try:
            person_name = pdir.name.replace('_', ' ').title()
            img_files = [f for f in pdir.glob('*.*') if f.suffix.lower().replace('.', '') in ALLOWED_EXTENSIONS]
            if not img_files:
                continue

            embeddings_list = []
            fingerprint_paths = []
            primary_b64 = ""
            for idx, img_path in enumerate(img_files[:10]):
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                features = matcher.extract_features(img_bytes)
                embeddings_list.append(features["embedding"])

                ext = img_path.suffix.lstrip('.')
                if idx == 0:
                    primary_b64 = f"data:image/{ext};base64,{base64.b64encode(img_bytes).decode('utf-8')}"

                dest_name = f"{secure_filename(person_name).lower()}_scan{idx+1}_{uuid.uuid4().hex[:6]}.{ext}"
                try:
                    dest_path = FINGERPRINT_FOLDER / dest_name
                    with open(dest_path, 'wb') as f:
                        f.write(img_bytes)
                    sample_path = SAMPLE_FOLDER / dest_name
                    with open(sample_path, 'wb') as f:
                        f.write(img_bytes)
                except Exception:
                    pass

                fingerprint_paths.append(f"/uploads/fingerprints/{dest_name}")

            user_record = {
                "id": f"usr_ds_{uuid.uuid4().hex[:8]}",
                "name": person_name,
                "mobile": "+1 (555) 000-0000",
                "blood_group": "O+",
                "emergency_contact": "Emergency Services",
                "address": f"Enrolled via Dataset ({pdir.name})",
                "age": 30,
                "gender": "Other",
                "fingerprint_path": fingerprint_paths[0] if fingerprint_paths else "",
                "fingerprint_image": primary_b64,
                "fingerprint_paths": fingerprint_paths,
                "embeddings": embeddings_list
            }
            db.insert_user(user_record)
            imported_count += 1
        except Exception as err:
            errors_list.append(f"Failed {pdir.name}: {str(err)}")

    return jsonify({
        "success": True,
        "imported_persons": imported_count,
        "errors": errors_list,
        "message": f"Successfully imported {imported_count} person(s) from dataset/."
    })


import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

if __name__ == '__main__':
    print(f"[Server] Starting Biometric Fingerprint Recognition Server on http://127.0.0.1:{PORT}")
    app.run(host='0.0.0.0', port=PORT, debug=DEBUG, use_reloader=False)

