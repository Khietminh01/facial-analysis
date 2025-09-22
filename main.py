import cv2
import argparse
import warnings
import numpy as np

from models import SCRFD, Attribute, Liveness
from utils.helpers import Face, draw_face_info

warnings.filterwarnings("ignore")


def load_models(detection_model_path: str, attribute_model_path: str, liveness_model_path: str):
    """Loads the detection, attribute, and liveness models."""
    try:
        detection_model = SCRFD(model_path=detection_model_path)
        attribute_model = Attribute(model_path=attribute_model_path)
        liveness_model = Liveness(model_path=liveness_model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    return detection_model, attribute_model, liveness_model


def inference_image(detection_model, attribute_model, liveness_model, image_path, save_output):
    """Processes a single image for face detection, attributes, and liveness."""
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image")
        return

    process_frame(detection_model, attribute_model, liveness_model, frame)
    if save_output:
        cv2.imwrite(save_output, frame)
    cv2.imshow("FaceDetection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inference_video(detection_model, attribute_model, liveness_model, video_source, save_output):
    """Processes a video source for face detection, attributes, and liveness."""
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Failed to open video source")
        return

    out = None
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(save_output, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(detection_model, attribute_model, liveness_model, frame)
        if save_output:
            out.write(frame)

                # Resize về size cố định 1920x1080 để hiển thị
        display_frame = cv2.resize(frame, (1920, 1080))
        cv2.imshow("FaceDetection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if save_output:
        out.release()
    cv2.destroyAllWindows()


def process_frame(detection_model, attribute_model, liveness_model, frame):
    """Detects faces, attributes, and liveness in a frame and draws the information."""
    boxes_list, points_list = detection_model.detect(frame)

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf_score = boxes
        gender, age = attribute_model.get(frame, bbox)
        liveness = liveness_model.get(frame, bbox)
        face = Face(kps=keypoints, bbox=bbox, age=age, gender=gender, liveness=liveness)
        draw_face_info(frame, face)


def run_face_analysis(detection_weights, attribute_weights, liveness_weights, input_source, save_output=None):
    """Runs face detection on the given input source."""
    detection_model, attribute_model, liveness_model = load_models(detection_weights, attribute_weights, liveness_weights)

    if isinstance(input_source, str) and input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
        inference_image(detection_model, attribute_model, liveness_model, input_source, save_output)
    else:
        inference_video(detection_model, attribute_model, liveness_model, input_source, save_output)


def main():
    """Main function to run face detection from command line."""
    parser = argparse.ArgumentParser(description="Run face detection on an image or video")
    parser.add_argument(
        '--detection-weights',
        type=str,
        default=r"C:\Users\minhk\Downloads\output\det_500m.onnx",
        help='Path to the detection model weights file'
    )
    parser.add_argument(
        '--attribute-weights',
        type=str,
        default=r"C:\Users\minhk\Downloads\output\genderage.onnx",
        help='Path to the attribute model weights file'
    )
    parser.add_argument(
        '--liveness-weights',
        type=str,
        default=r"C:\Users\minhk\Downloads\output\Liveness_80x80_MiniFASNetV1SE.onnx",
        help='Path to the liveness model weights file'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=r"C:\Users\minhk\Downloads\output\2025-09-18_07-51-21_Cam5.mp4",
        help='Path to the input image or video file or camera index (0, 1, ...)'
    )
    parser.add_argument('--output', type=str, help='Path to save the output image or video')
    args = parser.parse_args()

    run_face_analysis(args.detection_weights, args.attribute_weights, args.liveness_weights, args.source, args.output)


if __name__ == "__main__":
    main()
