import cv2
import numpy as np
import onnxruntime
from typing import Tuple

from utils.helpers import image_alignment

__all__ = ["Glass"]


class Glass:
    def __init__(self, model_path: str) -> None:
        """
        Glasses Classification
        Args:
            model_path (str): Path to .onnx file
        """
        self.model_path = model_path
        self.class_names = ["no_glasses", "glasses"]

        self.input_std = 1.0
        self.input_mean = 0.0
        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        metadata = self.session.get_inputs()[0]
        input_shape = metadata.shape  # [1, 224, 224, 3] hoặc [1,3,224,224]

        # Xác định (height, width) từ shape
        if len(input_shape) == 4:
            if input_shape[1] == 3:   # NCHW
                self.input_size = (input_shape[2], input_shape[3])
            else:                     # NHWC
                self.input_size = (input_shape[1], input_shape[2])
        else:
            self.input_size = (224, 224)

        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

        print("Input name:", self.input_names)
        print("Input shape:", input_shape)
        print("Output names:", self.output_names)

    def preprocess(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Cắt & resize khuôn mặt để đưa vào model"""
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        scale = self.input_size[0] / (max(width, height) * 1.5)
        transformed_image, _ = image_alignment(image, center, self.input_size[0], scale)

        resized = cv2.resize(transformed_image, self.input_size)  # (224,224,3)
        img = resized.astype(np.float32)
        img = (img - self.input_mean) / self.input_std

        # kiểm tra model cần NHWC hay NCHW
        if self.session.get_inputs()[0].shape[1] == 3:  # NCHW
            img = np.transpose(img, (2, 0, 1))  # (3,224,224)
        img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, predictions: np.ndarray) -> Tuple[int, float]:
        """
        Xử lý output của model:
        - Nếu model output 1 logit → dùng sigmoid
        - Nếu model output 2 logits → dùng softmax
        """
        if predictions.shape[0] == 1:  # single logit → sigmoid
            prob = 1 / (1 + np.exp(-predictions[0]))
            idx = int(prob >= 0.5)
            return idx, float(prob)
        else:  # 2 logits → softmax
            probs = self._softmax(predictions)
            idx = int(np.argmax(probs))
            return idx, float(probs[idx])

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def get(self, image: np.ndarray, bbox: np.ndarray) -> int:
        blob = self.preprocess(image, bbox)
        predictions = self.session.run(
            self.output_names,
            {self.input_names[0]: blob}
        )[0][0]

        idx, prob = self.postprocess(predictions)
        print(f"🔎 Glass raw={predictions}, prob={prob:.4f}, class={idx} ({self.class_names[idx]})")

        return idx
