# import the necessary packages
import numpy as np
import cv2
import os

try:
    import tf_keras
    from tf_keras.models import load_model
    TF_AVAILABLE = True
except ImportError:
    try:
        from tensorflow.keras.models import load_model
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False

from . import config


class PredictDisaster:
    def __init__(self):
        if TF_AVAILABLE:
            model_path = config.MODEL_PATH
            # If file doesn't have .h5 extension, check for .h5 version
            if not model_path.endswith('.h5') and os.path.exists(model_path + '.h5'):
                model_path = model_path + '.h5'
            elif not os.path.exists(model_path) and os.path.exists(model_path.replace('.model', '.h5')):
                model_path = model_path.replace('.model', '.h5')
            try:
                self.model = load_model(model_path)
                print(f"[INFO] Model loaded from {model_path}")
            except Exception as e:
                print(f"[WARNING] Could not load model: {e}")
                self.model = None
        else:
            self.model = None
        self.W = None
        self.H = None

    def predict(self, frame):
        if not TF_AVAILABLE or self.model is None:
            # Return a dummy prediction (no disaster) when TF is not available
            return np.array([0.2, 0.8])

        if self.W is None or self.H is None:
            (self.H, self.W) = frame.shape[:2]

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = frame.astype("float32")

        preds = self.model.predict(np.expand_dims(frame, axis=0), verbose=0)[0]

        return preds
