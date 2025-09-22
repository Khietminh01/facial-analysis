import cv2
import numpy as np
import onnxruntime
from typing import Tuple, Optional

__all__ = ["Liveness"]

class Liveness:
    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        """Liveness Detection Model
        Args:
            model_path (str): Path to .onnx file
            threshold (float): Minimum probability to accept prediction
        """
        self.model_path = model_path
        self.threshold = threshold
        self._initialize_model(model_path)

    def _initialize_model(self, model_path: str):
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            metadata = self.session.get_inputs()[0]
            input_shape = metadata.shape  # [N, C, H, W]
            self.input_size = (input_shape[3], input_shape[2])  # (W, H)
            self.input_names = [x.name for x in self.session.get_inputs()]
            self.output_names = [x.name for x in self.session.get_outputs()]
        except Exception as e:
            print(f"Failed to load liveness model: {e}")
            raise

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        h, w, _ = image.shape
        x1, y1, x2, y2 = map(int, bbox)

        bw, bh = x2 - x1, y2 - y1
        dw, dh = int(bw * 0.3 / 2), int(bh * 0.3 / 2)
        x1 = max(0, x1 - dw)
        y1 = max(0, y1 - dh)
        x2 = min(w, x2 + dw)
        y2 = min(h, y2 + dh)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return None

        face = cv2.resize(face, self.input_size)
        # >>> giữ nguyên BGR, KHÔNG chia 255 <<<
        face = face.astype(np.float32).transpose(2, 0, 1)  # HWC -> CHW
        face = np.expand_dims(face, axis=0)  # NCHW
        return face

    def postprocess(self, predictions: np.ndarray) -> Tuple[int, float]:
        logits = predictions[0]
        # >>> chuyển logits -> softmax probabilities <<<
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        idx = int(np.argmax(probs))
        conf = float(probs[idx])

        mapping = {0: -1, 1: 0, 2: 1}
        if conf < self.threshold:
            return -1, conf
        return mapping.get(idx, -1), conf

    def get(self, image: np.ndarray, bbox: np.ndarray) -> int:
        blob = self.preprocess(image, bbox)
        if blob is None:
            return -1
        preds = self.session.run(self.output_names, {self.input_names[0]: blob})[0]
        label, conf = self.postprocess(preds)
        return label
