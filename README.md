# 🧬 DEEPPRINT-ID — Deep Learning-Based Fingerprint Recognition for Unknown Person Identification

A biometric fingerprint recognition web application powered by **Deep Convolutional Neural Networks (CNN)**, **OpenCV Gabor/CLAHE Ridge Enhancement**, **MongoDB Atlas Database**, and a futuristic biometric terminal UI.

---

## 🌟 Key Features

1. **Instant Biometric Scan & Detection**:
   - Upload or drag-and-drop any scanned fingerprint image (PNG, JPG, BMP, TIF).
   - Accurately detects and displays complete citizen/user records:
     - **Full Legal Name**
     - **Mobile Number** (with 1-click copy/dial)
     - **Blood Group** (with highlighted crimson badge: O+, A+, B+, AB+, etc.)
     - **🚨 Emergency Contact Number & Relation** (with emergency alert button)
     - **Full Residential Address**
     - **Age & Gender**
     - **Minutiae Feature Count & CNN Latent Similarity Score**
   - High-tech holographic laser scanning animation with Web Audio API sound effects.

2. **Citizen Enrollment Studio**:
   - Register new individuals with complete personal records.
   - Upload fingerprint scan images with automated ridge quality assessment.
   - Computes 256-D deep CNN latent space embeddings and saves directly into MongoDB Atlas.

3. **MongoDB Atlas Database Integration with `.env` Configuration**:
   - Dedicated `.env` and `.env.example` files to configure your MongoDB Atlas cluster URI (`MONGODB_URI`).
   - Seamless failover & resilient local synchronization if Atlas connection is offline or pending.

4. **Deep CNN Architecture & Visualizer**:
   - Multi-scale Convolutional Neural Network with Batch Normalization, Dropout, Global Average Pooling, and Dense 256-D L2-normalized projection.
   - Live intermediate activation map visualizer (inspects ridge filters, minutiae kernels, core and delta patterns).

5. **Citizen Biometric Directory**:
   - Search by name, mobile, address, or blood group.
   - Filter chips for all blood groups (O+, A+, B+, AB+, etc.).
   - Printable ID Badge modal with QR/Barcode and CNN security hash.

6. **1-Click Test Samples Drawer**:
   - Pre-loaded with realistic sample users and fingerprints so you can test identification with a single click immediately upon launch.

---

## 🚀 Quick Start Guide

### 1. Configure MongoDB Atlas (`.env`)
Open `.env` and insert your MongoDB Atlas URI:
```env
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
DB_NAME=biometric_fingerprint_db
COLLECTION_NAME=users
PORT=5000
DEBUG=True
MATCH_THRESHOLD=0.65
```

### 2. Install Dependencies (if needed)
```bash
pip install -r requirements.txt
```

### 3. Run Pre-Seeded Dataset (Optional)
```bash
python seed_data.py
```

### 4. Start the Web Server
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 🏗️ Project Architecture

```
Fingerprint project 1/
├── .env                       # Environment variables (MongoDB Atlas URI, DB settings)
├── .env.example               # Template for environment configuration
├── app.py                     # Flask Web Server & REST API endpoints
├── cnn_model.py               # Deep CNN Biometric Embedding & Feature Extractor
├── config.py                  # Environment loader & path configurations
├── database.py                # MongoDB Atlas connector with PyMongo & fallback
├── fingerprint_matcher.py     # CLAHE, Gabor filtering, minutiae extraction & matcher
├── seed_data.py               # Synthetic realistic fingerprint generator & seed data
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html             # Responsive Biometric Terminal Dashboard
├── static/
│   ├── css/
│   │   └── styles.css         # Dark glassmorphism & biometric laser HUD styling
│   └── js/
│       └── app.js             # Web Audio SFX, scanning pipeline & directory controller
└── uploads/
    ├── fingerprints/          # Stored enrolled fingerprint scans
    └── samples/               # Pre-loaded 1-click test fingerprints
```
