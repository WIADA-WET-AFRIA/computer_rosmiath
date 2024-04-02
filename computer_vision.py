import cv2
from ultralytics import YOLO
import threading

# Ceci est un commentaire

# Définir les fichiers vidéo pour les trackers
video_file01 = 'ultralytics/test.mp4'
video_file02 = 0  # Chemin de la webcam (index)

# Charger les modèles YOLOv8
model1 = YOLO("yolov8n.pt")
model2 = YOLO("yolov8s.pt")

def run_tracker_in_thread(filename, model, file_index):
    video = cv2.VideoCapture(filename)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        # Suivre les objets dans le cadre s'ils sont disponibles
        results = model.track(frame, persist=True)
        res_plotted = results[0].plot()
        
        cv2.imshow(f"Tracking_{file_index}", res_plotted)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
    video.release()

tracer1 = threading.Thread(target=run_tracker_in_thread, args=(video_file01, model1, 1), daemon=True)
tracer2 = threading.Thread(target=run_tracker_in_thread, args=(video_file02, model2, 2), daemon=True)

tracer1.start()
tracer2.start()
tracer1.join()
tracer2.join()

cv2.destroyAllWindows()
