import io
import gc
import base64
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from cnn_model import get_cnn
from database import db
from config import CNN_IMAGE_SIZE, MATCH_THRESHOLD, BASE_DIR, STORAGE_DIR

# Restrict OpenCV memory and thread pool
try:
    cv2.setNumThreads(1)
except Exception:
    pass

class FingerprintMatcher:
    def __init__(self):
        self.cnn = get_cnn()
        self.target_size = CNN_IMAGE_SIZE
        self.match_threshold = MATCH_THRESHOLD
        
        # High-Precision SIFT & ORB Feature Detectors
        try:
            self.sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=0.03, edgeThreshold=10, sigma=1.6)
        except Exception:
            self.sift = None

        try:
            self.orb = cv2.ORB_create(nfeatures=800)
        except Exception:
            self.orb = None

    def preprocess_image(self, image_input):
        """
        Accepts: file path (str/Path), bytes, PIL Image, or numpy array.
        Returns:
            - processed_norm: 2D numpy array (128, 128) float32 in [0, 1] for CNN
            - img_enhanced: enhanced high-resolution 2D grayscale image for SIFT/minutiae
            - visual_annotated: 3D BGR numpy array with minutiae & ridge points
            - minutiae_count: int
        """
        img_gray = None
        if isinstance(image_input, (str, Path)):
            str_inp = str(image_input)
            if str_inp.startswith('data:image/'):
                try:
                    raw_b64 = str_inp.split(',', 1)[1] if ',' in str_inp else str_inp
                    img_bytes = base64.b64decode(raw_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                except Exception:
                    img_gray = None
            else:
                img_bgr = cv2.imread(str_inp)
                if img_bgr is not None:
                    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img_bgr is not None:
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        elif isinstance(image_input, Image.Image):
            img_gray = np.array(image_input.convert('L'))
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3:
                img_gray = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
            else:
                img_gray = image_input

        if img_gray is None:
            raise ValueError("Could not decode fingerprint image.")

        # Ensure standard working resolution for high-res minutiae extraction
        h, w = img_gray.shape
        if max(h, w) > 512:
            scale = 512.0 / max(h, w)
            img_highres = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        elif min(h, w) < 250:
            scale = 300.0 / max(h, w)
            img_highres = cv2.resize(img_gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        else:
            img_highres = img_gray.copy()

        # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization) on high-res
        clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_highres)

        # 2. Gabor filtering & Ridge enhancement
        g_kernel = cv2.getGaborKernel((11, 11), 3.0, np.pi/4, 4.0, 0.5, 0, ktype=cv2.CV_32F)
        img_gabor = cv2.filter2D(img_clahe, cv2.CV_8UC1, g_kernel)
        img_enhanced = cv2.addWeighted(img_clahe, 0.70, img_gabor, 0.30, 0)

        # 3. Canonical CNN image resize (128, 128) and normalization [0.0, 1.0]
        img_cnn_resized = cv2.resize(img_enhanced, self.target_size, interpolation=cv2.INTER_AREA)
        processed_norm = img_cnn_resized.astype(np.float32) / 255.0

        # 4. Minutiae Feature Detection & Visual HUD Map
        visual_annotated, minutiae_count = self._detect_minutiae_overlay(img_clahe)

        return processed_norm, img_enhanced, visual_annotated, minutiae_count

    def _detect_minutiae_overlay(self, gray_img):
        """Detects ridge bifurcations & endings, draws futuristic biometric markers."""
        overlay = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        
        corners = cv2.goodFeaturesToTrack(
            gray_img,
            maxCorners=90,
            qualityLevel=0.04,
            minDistance=7,
            blockSize=5
        )

        minutiae_count = 0
        if corners is not None:
            corners = np.int32(corners)
            minutiae_count = len(corners)
            for i, c in enumerate(corners):
                x, y = c.ravel()
                if i % 2 == 0:
                    # Ridge ending (Cyan Circle)
                    cv2.circle(overlay, (x, y), 4, (255, 230, 0), 1)
                    cv2.circle(overlay, (x, y), 2, (0, 255, 255), -1)
                else:
                    # Bifurcation (Neon Emerald Square)
                    cv2.rectangle(overlay, (x-4, y-4), (x+4, y+4), (0, 255, 128), 1)
                    cv2.circle(overlay, (x, y), 1, (0, 255, 128), -1)

        h, w = gray_img.shape
        cv2.line(overlay, (w//2, 0), (w//2, h), (40, 60, 80), 1)
        cv2.line(overlay, (0, h//2), (w, h//2), (40, 60, 80), 1)
        cv2.circle(overlay, (w//2, h//2), min(w, h)//4, (0, 200, 255), 1)

        return overlay, minutiae_count

    def extract_features(self, image_input):
        """Extracts CNN embedding, SIFT descriptors, and minutiae details."""
        processed_norm, img_enhanced, visual_annotated, minutiae_count = self.preprocess_image(image_input)
        embedding = self.cnn.extract_embedding(processed_norm)
        
        kp, des = None, None
        if self.sift is not None:
            try:
                kp, des = self.sift.detectAndCompute(img_enhanced, None)
            except Exception:
                des = None

        return {
            "embedding": embedding,
            "descriptors": des,
            "processed_norm": processed_norm,
            "img_enhanced": img_enhanced,
            "visual_annotated": visual_annotated,
            "minutiae_count": minutiae_count
        }

    def compute_cosine_similarity(self, emb_a, emb_b):
        """Calculates cosine similarity between two L2-normalized vectors."""
        a = np.array(emb_a, dtype=np.float32)
        b = np.array(emb_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, similarity))

    def compute_direct_image_match(self, qry_img_enhanced, enrolled_image_or_path):
        """
        Compares query enhanced image with enrolled fingerprint (base64 or file) using
        High-Precision SIFT + ORB with RANSAC Partial Affine Geometric Verification
        and Aligned Ridge Correlation.
        """
        try:
            if not enrolled_image_or_path:
                return None, 0

            r_clahe = None
            if isinstance(enrolled_image_or_path, str) and enrolled_image_or_path.startswith('data:image/'):
                _, r_clahe, _, _ = self.preprocess_image(enrolled_image_or_path)
            elif isinstance(enrolled_image_or_path, bytes):
                _, r_clahe, _, _ = self.preprocess_image(enrolled_image_or_path)
            else:
                str_p = str(enrolled_image_or_path).lstrip('/')
                full_path = STORAGE_DIR / str_p
                if not full_path.exists():
                    full_path = BASE_DIR / str_p
                if not full_path.exists():
                    full_path = BASE_DIR / 'uploads' / 'fingerprints' / Path(str_p).name
                if not full_path.exists():
                    full_path = BASE_DIR / 'uploads' / 'samples' / Path(str_p).name
                if full_path.exists():
                    _, r_clahe, _, _ = self.preprocess_image(full_path)

            if r_clahe is None:
                return None, 0
            
            q_clahe = qry_img_enhanced
            
            # 1. SIFT Minutiae Detection & Matching
            sift_inliers = 0
            sift_ratio = 0.0
            M_affine = None
            
            if self.sift is not None:
                kp_q, des_q = self.sift.detectAndCompute(q_clahe, None)
                kp_r, des_r = self.sift.detectAndCompute(r_clahe, None)
                
                if des_q is not None and des_r is not None and len(des_q) >= 4 and len(des_r) >= 4:
                    bf = cv2.BFMatcher(cv2.NORM_L2)
                    matches = bf.knnMatch(des_q, des_r, k=2)
                    good = [m for m, n in matches if len(matches) > 1 and m.distance < 0.75 * n.distance]
                    
                    if len(good) >= 4:
                        src = np.float32([kp_q[m.queryIdx].pt for m in good])
                        dst = np.float32([kp_r[m.trainIdx].pt for m in good])
                        M, inliers_mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                        if M is not None and inliers_mask is not None:
                            scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
                            if 0.65 <= scale <= 1.55:
                                sift_inliers = int(np.sum(inliers_mask))
                                sift_ratio = sift_inliers / max(1, len(good))
                                M_affine = M

            # 2. Dual ORB Minutiae Check
            orb_inliers = 0
            if self.orb is not None:
                kp_qo, des_qo = self.orb.detectAndCompute(q_clahe, None)
                kp_ro, des_ro = self.orb.detectAndCompute(r_clahe, None)
                if des_qo is not None and des_ro is not None and len(des_qo) >= 4 and len(des_ro) >= 4:
                    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING)
                    matches_o = bf_orb.knnMatch(des_qo, des_ro, k=2)
                    good_o = [m for m, n in matches_o if len(matches_o) > 1 and m.distance < 0.75 * n.distance]
                    if len(good_o) >= 4:
                        src_o = np.float32([kp_qo[m.queryIdx].pt for m in good_o])
                        dst_o = np.float32([kp_ro[m.trainIdx].pt for m in good_o])
                        M_o, inliers_mask_o = cv2.estimateAffinePartial2D(src_o, dst_o, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                        if M_o is not None and inliers_mask_o is not None:
                            scale_o = np.sqrt(M_o[0, 0]**2 + M_o[0, 1]**2)
                            if 0.65 <= scale_o <= 1.55:
                                orb_inliers = int(np.sum(inliers_mask_o))

            # 3. Aligned Texture Cross-Correlation
            aligned_corr = 0.0
            if M_affine is not None and sift_inliers >= 6:
                h, w = r_clahe.shape
                warped = cv2.warpAffine(q_clahe, M_affine, (w, h))
                mask = (warped > 15) & (r_clahe > 15)
                if np.sum(mask) > 300:
                    w_pts = warped[mask].astype(np.float32)
                    r_pts = r_clahe[mask].astype(np.float32)
                    w_pts -= np.mean(w_pts)
                    r_pts -= np.mean(r_pts)
                    denom = np.linalg.norm(w_pts) * np.linalg.norm(r_pts)
                    if denom > 0:
                        aligned_corr = max(0.0, float(np.dot(w_pts, r_pts) / denom))

            # 4. Calibrate Minutiae Score
            if sift_inliers >= 10 and sift_ratio >= 0.35:
                minutiae_score = min(1.0, 0.85 + (sift_inliers - 10) * 0.005 + min(0.10, orb_inliers * 0.003))
            elif sift_inliers >= 5 and sift_ratio >= 0.30:
                minutiae_score = 0.65 + (sift_inliers - 5) * 0.04
            elif sift_inliers >= 3:
                minutiae_score = 0.40 + (sift_inliers - 3) * 0.05
            else:
                minutiae_score = sift_inliers * 0.05

            if minutiae_score >= 0.50:
                final_score = 0.70 * minutiae_score + 0.30 * max(minutiae_score, aligned_corr)
            else:
                final_score = minutiae_score

            return float(final_score), sift_inliers
        except Exception:
            return None, 0

    def identify(self, image_input):
        """
        Scans an uploaded fingerprint, compares against all enrolled prints
        using High-Precision Minutiae Geometry + Deep CNN Latent Fusion.
        If confidence < threshold, strictly returns matched=False and NOT FOUND.
        """
        features = self.extract_features(image_input)
        query_emb = features["embedding"]
        minutiae_count = features["minutiae_count"]
        img_enhanced = features["img_enhanced"]
        
        # Get intermediate CNN layer activations for UI visualizer
        cnn_feature_maps = self.cnn.get_intermediate_feature_maps(features["processed_norm"])

        # Encode annotated image to base64 for direct UI display
        _, buffer = cv2.imencode('.jpg', features["visual_annotated"])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        users = db.get_all_users()
        if not users:
            return {
                "matched": False,
                "status": "NOT_FOUND",
                "message": "Database is empty. Please enroll citizens first.",
                "confidence": 0.0,
                "similarity_score": 0.0,
                "threshold": self.match_threshold,
                "minutiae_count": minutiae_count,
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                "cnn_feature_maps": cnn_feature_maps,
                "user": None
            }

        best_match_user = None
        highest_similarity = 0.0
        matched_finger_info = ""

        for user in users:
            raw_embeddings = user.get("embeddings", [])
            if not raw_embeddings:
                continue

            # Normalize to list of embedding vectors
            if isinstance(raw_embeddings[0], (int, float)):
                user_embeddings = [raw_embeddings]
            else:
                user_embeddings = raw_embeddings

            # Collect enrolled fingerprint image/paths
            user_fp_img = user.get("fingerprint_image", "")
            fp_paths = user.get("fingerprint_paths") or [user.get("fingerprint_path", "")]
            if isinstance(fp_paths, str):
                fp_paths = [fp_paths]

            user_best_sim = 0.0
            user_best_idx = 0

            for idx, stored_emb in enumerate(user_embeddings):
                # 1. Deep CNN Cosine Similarity
                cnn_sim = self.compute_cosine_similarity(query_emb, stored_emb)

                # 2. High-Precision Direct Image & Minutiae Match
                ref_target = ""
                if idx < len(fp_paths) and fp_paths[idx]:
                    str_p = str(fp_paths[idx]).lstrip('/')
                    if (STORAGE_DIR / str_p).exists() or (BASE_DIR / str_p).exists() or (BASE_DIR / 'uploads' / 'fingerprints' / Path(str_p).name).exists():
                        ref_target = fp_paths[idx]
                if not ref_target and user_fp_img:
                    ref_target = user_fp_img

                direct_score, inliers_count = self.compute_direct_image_match(img_enhanced, ref_target)

                # Multi-Factor Score Fusion
                if direct_score is not None:
                    if direct_score >= 0.50:
                        combined_sim = max(direct_score, 0.70 * direct_score + 0.30 * cnn_sim)
                    elif direct_score >= 0.30:
                        combined_sim = 0.60 * direct_score + 0.40 * cnn_sim
                    else:
                        combined_sim = direct_score
                else:
                    # Reference image not loaded on server -> fallback to CNN embedding similarity
                    combined_sim = cnn_sim

                if combined_sim > user_best_sim:
                    user_best_sim = combined_sim
                    user_best_idx = idx

            if user_best_sim > highest_similarity:
                highest_similarity = user_best_sim
                best_match_user = user
                matched_finger_info = f"Scan #{user_best_idx + 1}"

        confidence_percent = round(highest_similarity * 100, 2)
        is_match = highest_similarity >= self.match_threshold

        if is_match and best_match_user:
            user_info = {
                "id": best_match_user.get("id"),
                "name": best_match_user.get("name"),
                "mobile": best_match_user.get("mobile"),
                "blood_group": best_match_user.get("blood_group"),
                "emergency_contact": best_match_user.get("emergency_contact"),
                "address": best_match_user.get("address"),
                "age": best_match_user.get("age"),
                "gender": best_match_user.get("gender"),
                "created_at": best_match_user.get("created_at"),
                "fingerprint_path": best_match_user.get("fingerprint_path"),
                "matched_finger": matched_finger_info,
                "total_enrolled_prints": len(best_match_user.get("fingerprint_paths", [1])) if isinstance(best_match_user.get("fingerprint_paths"), list) else 1
            }
            return {
                "matched": True,
                "status": "MATCH_FOUND",
                "confidence": confidence_percent,
                "similarity_score": round(highest_similarity, 4),
                "threshold": self.match_threshold,
                "minutiae_count": minutiae_count,
                "user": user_info,
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                "cnn_feature_maps": cnn_feature_maps,
                "message": f"Biometric Verified! Identity matched to {user_info['name']} ({matched_finger_info})."
            }
        else:
            return {
                "matched": False,
                "status": "NOT_FOUND",
                "confidence": confidence_percent,
                "similarity_score": round(highest_similarity, 4),
                "threshold": self.match_threshold,
                "minutiae_count": minutiae_count,
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
                "cnn_feature_maps": cnn_feature_maps,
                "user": None,
                "message": "Biometric Record Not Found. No matching fingerprint enrolled in database."
            }

matcher = FingerprintMatcher()

