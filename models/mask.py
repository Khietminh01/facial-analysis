import cv2
import numpy as np
import onnxruntime
from typing import Tuple

from utils.helpers import image_alignment

__all__ = ["Mask"]


class Mask:
    def __init__(self, model_path: str) -> None:
        """
        Mask Detection
        Args:
            model_path (str): Path to .onnx file
        """
        self.model_path = model_path

        # Nhãn tham khảo (không trả trực tiếp, chỉ để debug/log)
        self.class_names = ["incorect_mask","with_mask","without_mask"]

        self.input_std = 1.0
        self.input_mean = 0.0
        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        metadata = self.session.get_inputs()[0]
        input_shape = metadata.shape  # [1, 224, 224, 3]
        # Lấy (height, width) đúng
        self.input_size = (input_shape[1], input_shape[2])  
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

        print("Input name:", self.session.get_inputs()[0].name)
        print("Input shape:", self.session.get_inputs()[0].shape)


    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Cắt & resize khuôn mặt để đưa vào model"""
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        scale = self.input_size[0] / (max(width, height) * 1.5)
        transformed_image, _ = image_alignment(image, center, self.input_size[0], scale)

        resized = cv2.resize(transformed_image, self.input_size)  # (224,224,3)
        img = resized.astype(np.float32)
        img = (img - self.input_mean) / self.input_std
        img = np.expand_dims(img, axis=0)  # (1,224,224,3)



        return img





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
        print("Preprocess output shape:", blob.shape)
        predictions = self.session.run(
            self.output_names,
            {self.input_names[0]: blob}
        )[0][0]
        return self.postprocess(predictions)
