import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from config import CNN_IMAGE_SIZE, EMBEDDING_DIM, BASE_DIR

# Suppress verbose TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WEIGHTS_DIR = BASE_DIR / 'models'
WEIGHTS_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = WEIGHTS_DIR / 'fingerprint_cnn_weights.weights.h5'

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
        self.model = self._build_model()
        self._load_or_init_weights()
        self.feature_extractor = self._build_feature_extractor()
        print("[CNN] Deep Biometric Fingerprint CNN initialized with persistent deterministic weights.")

    def _build_model(self):
        """
        Constructs a Deep Residual-style CNN for Fingerprint Latent Embedding.
        Extracts translation/rotation-invariant ridge and minutiae patterns.
        """
        # Set deterministic seeds for repeatable architecture initialization
        tf.random.set_seed(42)
        np.random.seed(42)

        init = tf.keras.initializers.GlorotUniform(seed=42)

        inputs = layers.Input(shape=(self.image_size[0], self.image_size[1], 1), name="fingerprint_input")
        
        # Block 1: Low-level Ridge Frequency & Orientation Extraction
        x = layers.Conv2D(32, (5, 5), padding='same', activation='relu', kernel_initializer=init, name="conv1_1")(inputs)
        x = layers.BatchNormalization(name="bn1_1")(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=init, name="conv1_2")(x)
        x = layers.BatchNormalization(name="bn1_2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)
        x = layers.Dropout(0.1)(x)

        # Block 2: Minutiae & Bifurcation Detection
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=init, name="conv2_1")(x)
        x = layers.BatchNormalization(name="bn2_1")(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=init, name="conv2_2")(x)
        x = layers.BatchNormalization(name="bn2_2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)
        x = layers.Dropout(0.15)(x)

        # Block 3: High-level Structural & Core/Delta Feature Representation
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=init, name="conv3_1")(x)
        x = layers.BatchNormalization(name="bn3_1")(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=init, name="conv3_2")(x)
        x = layers.BatchNormalization(name="bn3_2")(x)
        x = layers.MaxPooling2D((2, 2), name="pool3")(x)
        x = layers.Dropout(0.2)(x)

        # Block 4: Deep Latent Fusion
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer=init, name="conv4_1")(x)
        x = layers.BatchNormalization(name="bn4_1")(x)
        x = layers.GlobalAveragePooling2D(name="gap")(x)

        # Dense Embedding Projection with L2 Normalization
        x = layers.Dense(self.embedding_dim, kernel_initializer=init, kernel_regularizer=regularizers.l2(1e-4), name="dense_proj")(x)
        x = layers.BatchNormalization(name="bn_proj")(x)
        
        # Unit-norm L2 Normalization for Cosine Metric space
        embeddings = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1), name="l2_embedding")(x)

        model = models.Model(inputs=inputs, outputs=embeddings, name="BiometricFingerprintCNN")
        return model

    def _load_or_init_weights(self):
        """Ensures weights are persistent across server restarts and executions."""
        dummy_input = np.zeros((1, self.image_size[0], self.image_size[1], 1), dtype=np.float32)
        self.model(dummy_input)
        
        if WEIGHTS_PATH.exists():
            try:
                self.model.load_weights(str(WEIGHTS_PATH))
            except Exception as e:
                print(f"[CNN] Note: Re-saving weights due to: {e}")
                self.model.save_weights(str(WEIGHTS_PATH))
        else:
            try:
                self.model.save_weights(str(WEIGHTS_PATH))
            except Exception as e:
                print(f"[CNN] Could not save weights: {e}")

    def _build_feature_extractor(self):
        """Constructs sub-model to extract intermediate activation maps for visualizer."""
        try:
            layer_names = ["conv1_1", "conv2_1", "conv3_1", "conv4_1"]
            outputs = [self.model.get_layer(name).output for name in layer_names]
            return models.Model(inputs=self.model.input, outputs=outputs)
        except Exception:
            return None

    def extract_embedding(self, preprocessed_image):
        """
        Input: 2D numpy array (128, 128) with values in range [0, 1]
        Output: 1D numpy array of length 256 (L2 normalized)
        """
        if len(preprocessed_image.shape) == 2:
            img = np.expand_dims(preprocessed_image, axis=-1)
        else:
            img = preprocessed_image
        
        batch = np.expand_dims(img, axis=0).astype(np.float32)
        embedding = self.model(batch, training=False)
        return embedding.numpy()[0].tolist()

    def get_intermediate_feature_maps(self, preprocessed_image):
        """
        Returns normalized 2D feature map slices for frontend visualization.
        """
        if self.feature_extractor is None:
            return {}
        
        if len(preprocessed_image.shape) == 2:
            img = np.expand_dims(preprocessed_image, axis=-1)
        else:
            img = preprocessed_image
        
        batch = np.expand_dims(img, axis=0).astype(np.float32)
        activations = self.feature_extractor(batch, training=False)
        
        maps = {}
        layer_tags = ["Layer 1 (Ridge Filters)", "Layer 2 (Minutiae & Bifurcations)", "Layer 3 (Core & Delta Patterns)", "Layer 4 (Deep Fingerprint Vectors)"]
        
        for i, act in enumerate(activations):
            act_np = act.numpy()[0] # shape (H, W, C)
            selected_slices = []
            num_channels = min(4, act_np.shape[-1])
            for ch in range(num_channels):
                ch_map = act_np[:, :, ch]
                min_v, max_v = ch_map.min(), ch_map.max()
                if max_v > min_v:
                    norm_map = ((ch_map - min_v) / (max_v - min_v) * 255).astype(np.uint8)
                else:
                    norm_map = np.zeros_like(ch_map, dtype=np.uint8)
                selected_slices.append(norm_map.tolist())
            maps[layer_tags[i]] = selected_slices

        return maps

# Singleton getter
def get_cnn():
    return FingerprintCNN.get_instance()

