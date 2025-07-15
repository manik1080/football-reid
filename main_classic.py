import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
from sort.tracker import SortTracker


def main(args):
    model = YOLO(args.model)
    print("Model class names:")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error opening video file {args.video}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = 1.0 / fps

    tracker = SortTracker(max_age=30, min_hits=5, iou_threshold=0.2)

    out = None
    i = 1
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    begin_time = time.time()

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        i += 1
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)[0]

        detections = []
        for *box, score, cls in results.boxes.data.tolist():
            cls = int(cls)
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            area = w * h
            if (
                cls in [1, 2, 3] and  # goalkeeper, player, referee
                score >= args.conf
            ):
                detections.append([x1, y1, x2, y2, score, cls])
        detections = np.array(detections)

        tracks = tracker.update(detections, None)
        for track in tracks:
            x1, y1, x2, y2, track_id, cls, conf = track
            label = model.names.get(int(cls), 'unknown')
            conf_str = f"{conf:.2f}" if conf is not None else "N/A"
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {int(track_id)}: {label} Conf: {conf_str}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if out is not None:
            out.write(frame)
            if i == fps*(args.duration//1): break
        else:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if i % 30 == 0:
            end_time = time.time() - begin_time
            print(f"Processing {i} frames took {end_time:.2f} seconds")
            begin_time = time.time()

        end_time = time.time() - start_time
        if end_time < delay:
            time.sleep(delay - end_time)

    cap.release()
    if out is not None:
        out.release()
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO + SORT tracking test")
    parser.add_argument('--video', type=str, default='videos/15sec_input_720p.mp4',
                        help='Path to input video file')
    parser.add_argument('--model', type=str, default='models/best.pt',
                        help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.7,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video file')
    parser.add_argument('--duration', type=float, default=0,
                        help='If output, duration of video to output')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Verbosity')
    args = parser.parse_args()
    main(args)