import cv2
import argparse
import warnings
import numpy as np
import os
import time

from models import SCRFD, Attribute, Liveness, Emotion, Mask, Glass
from utils.helpers import Face, draw_face_info

warnings.filterwarnings("ignore")


def load_models(detection_model_path: str, attribute_model_path: str, liveness_model_path: str, emotion_model_path: str, mask_model_path: str, glass_model_path: str):
    """Loads the detection, attribute, emotions and liveness models."""
    try:
        detection_model = SCRFD(model_path=detection_model_path)
        attribute_model = Attribute(model_path=attribute_model_path)
        liveness_model = Liveness(model_path=liveness_model_path)
        emotion_model = Emotion(model_path=emotion_model_path)
        mask_model = Mask(model_path=mask_model_path)
        glass_model = Glass(model_path=glass_model_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        raise
    return detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model


def inference_image(detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model, image_path, save_output):
    """Processes a single image for face detection, attributes, and liveness."""
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load image")
        return

    process_frame(detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model, frame)
    if save_output:
        cv2.imwrite(save_output, frame)
        print(f"[INFO] Saved output image to {save_output}")

    cv2.imshow("FaceDetection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inference_video(detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model, video_source, save_output, out_video=False):
    """Processes a video source for face detection, attributes, and liveness."""
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Failed to open video source")
        return

    out = None
    if out_video and save_output:
        ext = os.path.splitext(save_output)[1].lower()
        if ext in ['.mp4', '.m4v']:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(save_output, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        print(f"[INFO] Writing output video to {save_output}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model, frame)
        
        if out is not None:
            out.write(frame)

                # Resize về size cố định 1920x1080 để hiển thị
        display_frame = cv2.resize(frame, (1920, 1080))
        cv2.imshow("FaceDetection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out is not None:
        out.release()
        print(f"[INFO] Saved output video to {save_output}")

    cv2.destroyAllWindows()


def process_frame(detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model, frame):
    """Detects faces, attributes, emotions and liveness in a frame and draws the information."""

    start = time.time()   # bắt đầu đo

    boxes_list, points_list = detection_model.detect(frame)

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf_score = boxes
        gender, age = attribute_model.get(frame, bbox)
        liveness = liveness_model.get(frame, bbox)
        emotion_idx,_ = emotion_model.get(frame, bbox)
        mask_idx,_ = mask_model.get(frame, bbox)
        glass_idx = glass_model.get(frame, bbox)

        print(
            f"[INFO] Face detected | "
            f"BBox: {bbox} | Confidence: {conf_score:.2f} | "
            f"Gender: {'Male' if gender == 1 else 'Female'} | "
            f"Age: {age} | "
            f"Liveness: {liveness} | "
            f"Emotion: {emotion_idx} |"
            f"Mask: {mask_idx} | " 
            f"Glass: {glass_idx}" 
        )
        face = Face(kps=keypoints, bbox=bbox, age=age, gender=gender, liveness=liveness, emotion=emotion_idx, mask=mask_idx, glass=glass_idx)
        draw_face_info(frame, face)

        end = time.time()   # kết thúc đo
        elapsed = (end - start) * 1000  # ms
        fps = 1000 / elapsed if elapsed > 0 else 0
        print(f"[PERF] Frame time: {elapsed:.2f} ms | {fps:.2f} FPS")


def run_face_analysis(detection_weights, attribute_weights, liveness_weights, emotion_weights, mask_weights, glass_weights, input_source, save_output=None, out_video=False):
    """Runs face detection on the given input source."""
    detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model = load_models(detection_weights, attribute_weights, liveness_weights, emotion_weights, mask_weights, glass_weights)

    # Nếu chưa truyền save_output thì tự tạo
    if save_output is None:
        if isinstance(input_source, str) and input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
            base, ext = os.path.splitext(input_source)
            save_output = base + "_out" + ext
        elif isinstance(input_source, str):
            base, ext = os.path.splitext(input_source)
            save_output = base + "_out" + ext   # giữ nguyên đuôi video
        else:
            save_output = "camera_out.avi"
        print(f"[INFO] Auto output path: {save_output}")

    # Nếu input là video thì tự bật out_video nếu chưa bật
    if not input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
        out_video = True

    if isinstance(input_source, str) and input_source.lower().endswith(('.jpg', '.png', '.jpeg')):
        inference_image(detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model, input_source, save_output)
    else:
        inference_video(detection_model, attribute_model, liveness_model, emotion_model, mask_model, glass_model, input_source, save_output, out_video)


def main():
    """Main function to run face detection from command line."""
    parser = argparse.ArgumentParser(description="Run face detection on an image or video")
    parser.add_argument(
        '--detection-weights',
        type=str,
        default=r"C:\Users\ADMIN\Downloads\resources\resources\det_500m.onnx",
        help='Path to the detection model weights file'
    )
    parser.add_argument(
        '--attribute-weights',
        type=str,
        default=r"C:\Users\ADMIN\Downloads\resources\resources\genderage.onnx",
        help='Path to the attribute model weights file'
    )
    parser.add_argument(
        '--liveness-weights',
        type=str,
        default=r"C:\Users\ADMIN\Downloads\resources\resources\Liveness_80x80_MiniFASNetV1SE.onnx",
        help='Path to the liveness model weights file'
    )
    parser.add_argument(
        '--emotion-weights',
        type=str,
        default=r"C:\Users\ADMIN\Downloads\resources\resources\enet_b0_8_best_afew_emotion.onnx",
        help='Path to the emotions model weights file'
    )
    parser.add_argument(
        '--mask-weights',
        type=str,
        default=r"C:\Users\ADMIN\Downloads\resources\resources\mobilenetV2_224.onnx",
        help='Path to the mask model weights file'
    )
    parser.add_argument(
        '--glass-weights',
        type=str,
        default=r"C:\test\glasses_classifier.onnx",
        help='Path to the glass model weights file'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=r"C:\Users\ADMIN\Videos\Bản ghi màn hình\Quay màn hình 2025-09-24 005135.mp4",
        help='Path to the input image or video file or camera index (0, 1, ...)'
    )
    parser.add_argument('--output', type=str, help='Path to save the output image or video')
    args = parser.parse_args()

    run_face_analysis(args.detection_weights, args.attribute_weights, args.liveness_weights, args.emotion_weights, args.mask_weights, args.glass_weights, args.source, args.output)


if __name__ == "__main__":
    main()
