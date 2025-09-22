import cv2
import numpy as np
import onnxruntime
from typing import Tuple

__all__ = ["Liveness"]

class Liveness:
    def __init__(self, model_path: str) -> None:
        """Liveness Detection Model
        Args:
            model_path (str): Path to .onnx file
        """
        self.model_path = model_path
        self._initialize_model(model_path)

    def _initialize_model(self, model_path: str):
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            metadata = self.session.get_inputs()[0]
            input_shape = metadata.shape  # [N, C, H, W]
            self.input_size = tuple(input_shape[2:4][::-1])  # (W, H)
            self.input_names = [x.name for x in self.session.get_inputs()]
            self.output_names = [x.name for x in self.session.get_outputs()]
        except Exception as e:
            print(f"Failed to load liveness model: {e}")
            raise

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Crop + resize face region
        Args:
            image (np.ndarray): Input image (BGR)
            bbox (np.ndarray): [x1, y1, x2, y2]
        Returns:
            np.ndarray: Preprocessed blob
        """
        x1, y1, x2, y2 = map(int, bbox)
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return None

        face = cv2.resize(face, self.input_size)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face = face.transpose(2, 0, 1)  # HWC -> CHW
        face = np.expand_dims(face, axis=0)  # NCHW
        face /= 255.0  # normalize [0,1]
        return face

    def postprocess(self, predictions: np.ndarray) -> int:
        """Convert logits to class index
        Args:
            predictions (np.ndarray): [1, 3]
        Returns:
            int: liveness label (-1=NG, 0=Fake, 1=Real)
        """
        idx = int(np.argmax(predictions, axis=1)[0])
        mapping = {0: -1, 1: 0, 2: 1}  # map theo yêu cầu của bạn
        return mapping.get(idx, -1)

    def get(self, image: np.ndarray, bbox: np.ndarray) -> int:
        blob = self.preprocess(image, bbox)
        if blob is None:
            return -1
        preds = self.session.run(self.output_names, {self.input_names[0]: blob})[0]
        return self.postprocess(preds)
