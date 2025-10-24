import cv2
import argparse
import warnings
import numpy as np
import os
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
        print(f"[INFO] Saved output image to {save_output}")

    cv2.imshow("FaceDetection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def inference_video(detection_model, attribute_model, liveness_model, video_source, save_output, out_video=False):
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

        process_frame(detection_model, attribute_model, liveness_model, frame)

        if out is not None:
            out.write(frame)
        display_frame = cv2.resize(frame, (1520, 1080))
        cv2.imshow("FaceDetection", display_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out is not None:
        out.release()
        print(f"[INFO] Saved output video to {save_output}")
    cv2.destroyAllWindows()


def process_frame(detection_model, attribute_model, liveness_model, frame):
    """Detects faces, attributes, and liveness in a frame and draws the information."""
    boxes_list, points_list = detection_model.detect(frame)

    for boxes, keypoints in zip(boxes_list, points_list):
        *bbox, conf_score = boxes
        gender, age = attribute_model.get(frame, bbox)

        liveness = liveness_model.get(frame, bbox)

        print(
            f"[INFO] Face detected | "
            f"BBox: {bbox} | Confidence: {conf_score:.2f} | "
            f"Gender: {'Male' if gender == 1 else 'Female'} | "
            f"Age: {age} | "
            f"Liveness: {liveness}"
        )

        face = Face(kps=keypoints, bbox=bbox, age=age, gender=gender, liveness=liveness)
        draw_face_info(frame, face)


def run_face_analysis(detection_weights, attribute_weights, liveness_weights, input_source, save_output=None, out_video=False):
    """Runs face detection on the given input source."""
    detection_model, attribute_model, liveness_model = load_models(detection_weights, attribute_weights, liveness_weights)

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
        inference_image(detection_model, attribute_model, liveness_model, input_source, save_output)
    else:
        inference_video(detection_model, attribute_model, liveness_model, input_source, save_output, out_video)


def main():
    """Main function to run face detection from command line."""
    parser = argparse.ArgumentParser(description="Run face detection on an image or video")
    parser.add_argument(
        '--detection-weights',
        type=str,
        default=r"D:\AIBOX\insightFace\resources\det_500m.onnx",
        help='Path to the detection model weights file'
    )
    parser.add_argument(
        '--attribute-weights',
        type=str,
        default=r"D:\AIBOX\insightFace\resources\genderage.onnx",
        help='Path to the attribute model weights file'
    )
    parser.add_argument(
        '--liveness-weights',
        type=str,
        default=r"D:\AIBOX\insightFace\resources\2.7_80x80_MiniFASNetV2.onnx",
        help='Path to the liveness model weights file'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=r"D:\AIBOX\insightFace\resources\screen1.mp4",
        help='Path to the input image or video file or camera index (0, 1, ...)'
    )
    parser.add_argument('--output', type=str, help='Path to save the output image or video (auto-generated if not provided)')
    args = parser.parse_args()

    run_face_analysis(
        args.detection_weights,
        args.attribute_weights,
        args.liveness_weights,
        args.source,
        args.output
    )


if __name__ == "__main__":
    main()
    # # --- Khởi tạo model ---
    # detector = SCRFD(model_path=r"D:\AIBOX\insightFace\resources\det_500m.onnx")

    # # --- Folder input/output ---
    # input_videos = r"D:\AIBOX\insightFace\resources\archive\files"
    # output_faces = input_videos
    # os.makedirs(output_faces, exist_ok=True)

    # # --- Đếm ảnh crop ---
    # counter = 1

    # # --- Duyệt qua tất cả video trong folder ---
    # for vid_name in os.listdir(input_videos):
    #     if not vid_name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
    #         continue  # bỏ qua file không phải video

    #     vid_path = os.path.join(input_videos, vid_name)
    #     print(f"[INFO] Processing video: {vid_path}")

    #     cap = cv2.VideoCapture(vid_path)
    #     frame_id = 0
    #     frame_skip = 10
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret:
    #             break
    #         if frame_id % frame_skip != 0:
    #             frame_id += 1
    #             continue  # bỏ qua frame này

    #         frame_id += 1
    #         h, w = frame.shape[:2]
    #         # --- Detect mặt ---
    #         boxes_list, points_list = detector.detect(frame)

    #         for boxes in boxes_list:
    #             x1, y1, x2, y2, score = boxes.astype(np.int32)

    #             x1 = max(0, int(x1 * 0.95))
    #             y1 = max(0, int(y1 * 0.95))
    #             x2 = min(w, int(x2 * 1.05))
    #             y2 = min(h, int(y2 * 1.05))

    #             bw, bh = x2 - x1, y2 - y1
    #             side = max(bw, bh)

    #             # Lấy tâm bbox
    #             cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    #             # Tính lại x1, y1, x2, y2 cho vuông
    #             x1_new = max(0, cx - side // 2)
    #             y1_new = max(0, cy - side // 2)
    #             x2_new = min(w, cx + side // 2)
    #             y2_new = min(h, cy + side // 2)

    #             # --- Crop mặt (chú ý đúng thứ tự: [row, col] = [y, x]) ---
    #             face = frame[y1_new:y2_new, x1_new:x2_new]
    #             if face.size == 0:
    #                 continue

    #             # Lưu file theo dạng replay_0001.jpg
    #             vid_name_no_ext = os.path.splitext(vid_name)[0]  # Lấy tên video không có đuôi
    #             # Lưu file theo dạng tên_video_0001.jpg
    #             save_path = os.path.join(output_faces, f"{vid_name_no_ext}_{counter:04d}.jpg")
    #             cv2.imwrite(save_path, face)
    #             counter += 1

    #     cap.release()
    #     print(f"[INFO] Done video: {vid_name}")
    # print("[INFO] Finished! All faces saved in:", output_faces)
    # Test git


