# -*- coding: utf-8 -*-
"""
main_threaded_fixed.py

YOLO + SORT multi-threaded tracking with real-time display and optional recording.

Features:
  - Processes every frame in real-time mode (no --duration or duration=0).
  - Stride-based processing when recording a summary (--output with --duration > 0).
  - Proper synchronization to source FPS via cv2.waitKey.
  - Conditional VideoWriter: only active when recording.

Usage:
  # Real-time display (no output file):
  python main_threaded_fixed.py --video videos/15sec_input_720p.mp4 --model models/best.pt

  # Record a 10-second summary at 2x stride:
  python main_threaded_fixed.py --video videos/15sec_input_720p.mp4 --model models/best.pt \
      --output output.mp4 --duration 10 --stride 2
"""
import argparse
import time
import cv2
import numpy as np
from ultralytics import YOLO
from sort.tracker import SortTracker
import threading
from queue import Queue
from deep_sort_realtime.deepsort_tracker import DeepSort


def interpolate_tracks(track_a, track_b, alpha):
    interp = []
    id_map = {int(t[4]): t for t in track_a}
    for t2 in track_b:
        tid = int(t2[4])
        if tid in id_map:
            t1 = id_map[tid]
            x1 = (1 - alpha) * t1[0] + alpha * t2[0]
            y1 = (1 - alpha) * t1[1] + alpha * t2[1]
            x2 = (1 - alpha) * t1[2] + alpha * t2[2]
            y2 = (1 - alpha) * t1[3] + alpha * t2[3]
            interp.append([x1, y1, x2, y2, tid, t2[5], t2[6]])
    return interp


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
    delay_ms = int(1000 / fps)

    # Only record summary when output + positive duration
    do_record = bool(args.output and args.duration > 0)
    stride = args.stride if do_record else 1

    # Setup VideoWriter if recording
    out = None
    if do_record:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    tracker = DeepSort(max_age=10,
                      n_init=10,
                      max_cosine_distance=0.9,
                      max_iou_distance=0.7)
                      #nn_budget=100)

    frame_q = Queue(maxsize=20)
    result_q = Queue(maxsize=20)
    stop_flag = threading.Event()

    def reader():
        idx = 0
        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                frame_q.put((None, None))
                break
            frame_q.put((idx, frame))
            idx += 1

    def writer():
        while not stop_flag.is_set():
            idx, frame = result_q.get()
            if frame is None:
                break
            if do_record:
                if out is not None:
                    out.write(frame)
            else:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                    stop_flag.set()
                    # Put a sentinel in frame_q to unblock main loop
                    frame_q.put((None, None))
                    break

    reader_t = threading.Thread(target=reader, daemon=True)
    writer_t = threading.Thread(target=writer, daemon=True)
    reader_t.start()
    writer_t.start()

    last_idx = None
    last_tracks = None
    processed = {}
    frame_count = 0
    start_time = time.time()

    while True:
        idx, frame = frame_q.get()
        if frame is None or stop_flag.is_set():
            break

        if idx % stride == 0:
            # detect
            res = model(frame, conf=args.conf, verbose=False)[0]
            dets = []
            for *box, score, cls in res.boxes.data.tolist():
                cls = int(cls)
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                area = w * h
                if cls in [1, 2, 3] and score >= args.conf and area >= 400:
                    dets.append(([x1, y1, w, h], score, cls))
            # track
            tracks = tracker.update_tracks(dets, frame=frame)
            processed[idx] = (tracks, frame.copy())
            disp = frame.copy()
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cls = track.det_class
                label = model.names.get(cls, 'unknown')
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0,255,0),2)
                cv2.putText(disp, f"ID {track_id} {label}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
            result_q.put((idx, disp))

            # Interpolation for DeepSort tracks is non-trivial and would require matching track IDs between frames.
            # For now, you may skip interpolation or implement a similar logic if needed.

            last_idx = idx
            last_tracks = tracks

        frame_count += 1
        if do_record and frame_count >= fps * args.duration:
            break
        if stop_flag.is_set():
            import sys
            sys.exit(0)

    stop_flag.set()
    result_q.put((None, None))
    reader_t.join()
    writer_t.join()
    cap.release()
    if out:
        out.release()
    else:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="YOLO + SORT threaded real-time/tracking")
    parser.add_argument('--video',  type=str, default='videos/15sec_input_720p.mp4')
    parser.add_argument('--model',  type=str, default='models/best.pt')
    parser.add_argument('--conf',   type=float, default=0.7)
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video (requires --duration)')
    parser.add_argument('--duration', type=float, default=0,
                        help='Seconds to record (0 for live display)')
    parser.add_argument('--stride',  type=int, default=2,
                        help='Process every Nth frame in recording mode')
    args = parser.parse_args()
    main(args)

