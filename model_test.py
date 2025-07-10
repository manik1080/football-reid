import cv2
from ultralytics import YOLO


model = YOLO('models/best.pt')
video_path = 'videos/15sec_input_720p.mp4'
cap = cv2.VideoCapture(video_path)


class_names = set()
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output/model_output_test.mp4', fourcc, 30.0, (1280, 720))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model(frame)[0]

    for result in results.boxes:
        x1, y1, x2, y2 = result.xyxy[0].tolist()
        conf = float(result.conf[0])
        cls = int(result.cls[0])

        class_names.add(cls)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f'Class: {cls}, Conf: {conf:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    out.write(frame)
    cv2.imshow('Model Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or frame_count > 450:  # ~15s * 30fps
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Class labels detected in the video:")
print(sorted(class_names))
