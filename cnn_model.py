import os
from pathlib import Path
import numpy as np
from config import CNN_IMAGE_SIZE, EMBEDDING_DIM, BASE_DIR

WEIGHTS_DIR = BASE_DIR / 'models'
WEIGHTS_NPZ_PATH = WEIGHTS_DIR / 'fingerprint_cnn_weights.npz'

def _relu(x):
    return np.maximum(0, x)

def _batch_norm(x, gamma, beta, mean, var, eps=1e-3):
    inv_std = 1.0 / np.sqrt(var + eps)
    return (x - mean) * inv_std * gamma + beta

def _max_pool_2x2(x):
    H, W, C = x.shape
    return x.reshape(H//2, 2, W//2, 2, C).max(axis=(1, 3))

def _conv2d_same(x, kernel, bias):
    H, W, in_c = x.shape
    kH, kW, _, out_c = kernel.shape
    pad_h = kH // 2
    pad_w = kW // 2
    x_padded = np.pad(x, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    
    out_H, out_W = H, W
    cols = np.zeros((out_H * out_W, kH * kW * in_c), dtype=np.float32)
    col_idx = 0
    for i in range(kH):
        for j in range(kW):
            cols[:, col_idx * in_c:(col_idx + 1) * in_c] = x_padded[i:i+out_H, j:j+out_W, :].reshape(-1, in_c)
            col_idx += 1
            
    w_mat = kernel.reshape(-1, out_c)
    out = cols @ w_mat + bias
    return out.reshape(out_H, out_W, out_c)

class FingerprintCNN:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = FingerprintCNN()
        return cls._instance

    def __init__(self):
        self.image_size = CNN_IMAGE_SIZE
        self.embedding_dim = EMBEDDING_DIM
        self.weights = self._load_weights()
        print("[CNN Engine] Serverless-Optimized Deep Biometric CNN loaded (Pure NumPy/SciPy - Zero TensorFlow Overhead).")

    def _load_weights(self):
        if not WEIGHTS_NPZ_PATH.exists():
            raise FileNotFoundError(f"Weights file not found at {WEIGHTS_NPZ_PATH}")
        return np.load(WEIGHTS_NPZ_PATH)

    def extract_embedding(self, preprocessed_image):
        """
        Input: 2D numpy array (128, 128) float32 in [0, 1]
        Output: 1D list of length 256 (L2 normalized latent vector)
        """
        if len(preprocessed_image.shape) == 2:
            img = preprocessed_image
        else:
            img = preprocessed_image.squeeze()

        x = img[:, :, np.newaxis].astype(np.float32)
        
        # Block 1
        x = _conv2d_same(x, self.weights['conv1_1_0'], self.weights['conv1_1_1'])
        x = _batch_norm(x, self.weights['bn1_1_0'], self.weights['bn1_1_1'], self.weights['bn1_1_2'], self.weights['bn1_1_3'])
        x = _relu(x)
        
        x = _conv2d_same(x, self.weights['conv1_2_0'], self.weights['conv1_2_1'])
        x = _batch_norm(x, self.weights['bn1_2_0'], self.weights['bn1_2_1'], self.weights['bn1_2_2'], self.weights['bn1_2_3'])
        x = _relu(x)
        x = _max_pool_2x2(x)
        
        # Block 2
        x = _conv2d_same(x, self.weights['conv2_1_0'], self.weights['conv2_1_1'])
        x = _batch_norm(x, self.weights['bn2_1_0'], self.weights['bn2_1_1'], self.weights['bn2_1_2'], self.weights['bn2_1_3'])
        x = _relu(x)
        
        x = _conv2d_same(x, self.weights['conv2_2_0'], self.weights['conv2_2_1'])
        x = _batch_norm(x, self.weights['bn2_2_0'], self.weights['bn2_2_1'], self.weights['bn2_2_2'], self.weights['bn2_2_3'])
        x = _relu(x)
        x = _max_pool_2x2(x)
        
        # Block 3
        x = _conv2d_same(x, self.weights['conv3_1_0'], self.weights['conv3_1_1'])
        x = _batch_norm(x, self.weights['bn3_1_0'], self.weights['bn3_1_1'], self.weights['bn3_1_2'], self.weights['bn3_1_3'])
        x = _relu(x)
        
        x = _conv2d_same(x, self.weights['conv3_2_0'], self.weights['conv3_2_1'])
        x = _batch_norm(x, self.weights['bn3_2_0'], self.weights['bn3_2_1'], self.weights['bn3_2_2'], self.weights['bn3_2_3'])
        x = _relu(x)
        x = _max_pool_2x2(x)
        
        # Block 4
        x = _conv2d_same(x, self.weights['conv4_1_0'], self.weights['conv4_1_1'])
        x = _batch_norm(x, self.weights['bn4_1_0'], self.weights['bn4_1_1'], self.weights['bn4_1_2'], self.weights['bn4_1_3'])
        x = _relu(x)
        
        # Global Average Pooling
        gap = x.mean(axis=(0, 1))
        
        # Dense projection
        proj = gap @ self.weights['dense_proj_0'] + self.weights['dense_proj_1']
        proj = _batch_norm(proj, self.weights['bn_proj_0'], self.weights['bn_proj_1'], self.weights['bn_proj_2'], self.weights['bn_proj_3'])
        
        # L2 Normalize
        norm = np.linalg.norm(proj)
        if norm > 1e-12:
            embedding = proj / norm
        else:
            embedding = proj
            
        return embedding.tolist()

    def get_intermediate_feature_maps(self, preprocessed_image):
        """Returns normalized 2D feature map activations for UI visualization."""
        try:
            if len(preprocessed_image.shape) == 2:
                img = preprocessed_image
            else:
                img = preprocessed_image.squeeze()

            x = img[:, :, np.newaxis].astype(np.float32)
            
            # Layer 1
            x = _conv2d_same(x, self.weights['conv1_1_0'], self.weights['conv1_1_1'])
            x = _batch_norm(x, self.weights['bn1_1_0'], self.weights['bn1_1_1'], self.weights['bn1_1_2'], self.weights['bn1_1_3'])
            x = _relu(x)
            act1 = x
            
            x = _conv2d_same(x, self.weights['conv1_2_0'], self.weights['conv1_2_1'])
            x = _batch_norm(x, self.weights['bn1_2_0'], self.weights['bn1_2_1'], self.weights['bn1_2_2'], self.weights['bn1_2_3'])
            x = _relu(x)
            x = _max_pool_2x2(x)
            
            # Layer 2
            x = _conv2d_same(x, self.weights['conv2_1_0'], self.weights['conv2_1_1'])
            x = _batch_norm(x, self.weights['bn2_1_0'], self.weights['bn2_1_1'], self.weights['bn2_1_2'], self.weights['bn2_1_3'])
            x = _relu(x)
            act2 = x
            
            x = _conv2d_same(x, self.weights['conv2_2_0'], self.weights['conv2_2_1'])
            x = _batch_norm(x, self.weights['bn2_2_0'], self.weights['bn2_2_1'], self.weights['bn2_2_2'], self.weights['bn2_2_3'])
            x = _relu(x)
            x = _max_pool_2x2(x)
            
            # Layer 3
            x = _conv2d_same(x, self.weights['conv3_1_0'], self.weights['conv3_1_1'])
            x = _batch_norm(x, self.weights['bn3_1_0'], self.weights['bn3_1_1'], self.weights['bn3_1_2'], self.weights['bn3_1_3'])
            x = _relu(x)
            act3 = x
            
            x = _conv2d_same(x, self.weights['conv3_2_0'], self.weights['conv3_2_1'])
            x = _batch_norm(x, self.weights['bn3_2_0'], self.weights['bn3_2_1'], self.weights['bn3_2_2'], self.weights['bn3_2_3'])
            x = _relu(x)
            x = _max_pool_2x2(x)
            
            # Layer 4
            x = _conv2d_same(x, self.weights['conv4_1_0'], self.weights['conv4_1_1'])
            x = _batch_norm(x, self.weights['bn4_1_0'], self.weights['bn4_1_1'], self.weights['bn4_1_2'], self.weights['bn4_1_3'])
            x = _relu(x)
            act4 = x

            activations = [act1, act2, act3, act4]
            maps = {}
            layer_tags = ["Layer 1 (Ridge Filters)", "Layer 2 (Minutiae & Bifurcations)", "Layer 3 (Core & Delta Patterns)", "Layer 4 (Deep Fingerprint Vectors)"]
            
            for i, act_np in enumerate(activations):
                selected_slices = []
                num_channels = min(4, act_np.shape[-1])
                for ch in range(num_channels):
                    ch_map = act_np[:, :, ch]
                    if ch_map.shape[0] > 32 or ch_map.shape[1] > 32:
                        step_y = max(1, ch_map.shape[0] // 32)
                        step_x = max(1, ch_map.shape[1] // 32)
                        ch_map = ch_map[::step_y, ::step_x]
                    
                    min_v, max_v = float(ch_map.min()), float(ch_map.max())
                    if max_v > min_v:
                        norm_map = ((ch_map - min_v) / (max_v - min_v) * 255.0).astype(np.uint8)
                    else:
                        norm_map = np.zeros_like(ch_map, dtype=np.uint8)
                    selected_slices.append(norm_map.tolist())
                maps[layer_tags[i]] = selected_slices

            return maps
        except Exception:
            return {}

def get_cnn():
    return FingerprintCNN.get_instance()
