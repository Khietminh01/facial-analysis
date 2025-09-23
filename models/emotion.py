import cv2
import numpy as np
import onnxruntime
from typing import Tuple

from utils.helpers import image_alignment

__all__ = ["Emotion"]


class Emotion:
    def __init__(self, model_path: str) -> None:
        """
        Emotion Prediction
        Args:
            model_path (str): Path to .onnx file
        """
        self.model_path = model_path

        # Danh sách nhãn (không trả về trực tiếp, chỉ để tham khảo/debug)
        self.class_names = [
            "Angry", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"
        ]

        self.input_std = 1.0
        self.input_mean = 0.0
        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        """Khởi tạo ONNX runtime session"""
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        metadata = self.session.get_inputs()[0]
        input_shape = metadata.shape
        self.input_size = tuple(input_shape[2:4][::-1])  # (width, height)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Cắt & resize khuôn mặt để đưa vào model"""
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        scale = self.input_size[0] / (max(width, height) * 1.5)
        transformed_image, _ = image_alignment(image, center, self.input_size[0], scale)
        input_size = tuple(transformed_image.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(
            transformed_image,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        return blob

    def postprocess(self, predictions: np.ndarray) -> Tuple[int, float]:
        """Softmax + chọn class có xác suất cao nhất"""
        probs = self._softmax(predictions)
        idx = int(np.argmax(probs))

        return idx, float(probs[idx])

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[int, float]:
        """Trả về (index, score)"""
        blob = self.preprocess(image, bbox)
        predictions = self.session.run(
            self.output_names,
            {self.input_names[0]: blob}
        )[0][0]
        return self.postprocess(predictions)
