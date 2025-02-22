from ultralytics import YOLO
import cv2
import time

# Загрузка модели YOLOv8 (предобученной на COCO)
model = YOLO("yolov8n.pt")


prev_time = time.time()
# Открытие видеопотока с камеры (или укажите путь к видеофайлу)
capture = cv2.VideoCapture("3.mp4")  # Для видео используйте cv2.VideoCapture('video.mp4')

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Запуск модели YOLO на кадре
    results = model(frame)

    # Отображение результатов
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты рамки
            label = result.names[int(box.cls[0])]  # Имя класса
            conf = box.conf[0]  # Уверенность

            if label == "car":  # Фильтруем только автомобили
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            elif label == "truck":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Car Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()