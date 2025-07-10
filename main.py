import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


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

    tracker = DeepSort(max_age=10,
                       n_init=7,
                       max_cosine_distance=0.5,
                       nn_budget=None)

    out = None
    i = 0
    if args.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        begin_time = time.time()
        start_time = time.time()
        ret, frame = cap.read()
        if i % 30 == 0:
            end_time = time.time() - begin_time
            print(f"Processing {i} frames took {end_time:.2f} seconds")
            begin_time = time.time()
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
                score >= args.conf and
                area >= 800
            ):
                detections.append(([x1, y1, w, h], score, cls))

        tracks = tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cls = track.det_class
            conf = track.det_conf
            label = model.names.get(cls, 'unknown')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}: {label} Conf: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if out is not None:
            out.write(frame)
            if i == 120: break
        else:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = time.time() - start_time
        if end_time < delay:
            time.sleep(delay - end_time)

    cap.release()
    if out is not None:
        out.release()
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO + DeepSORT tracking test")
    parser.add_argument('--video', type=str, default='videos/15sec_input_720p.mp4',
                        help='Path to input video file')
    parser.add_argument('--model', type=str, default='models/best.pt',
                        help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.7,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video file')
    args = parser.parse_args()
    main(args)
