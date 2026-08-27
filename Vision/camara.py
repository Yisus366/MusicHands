import cv2
import mediapipe as mp
import time
from pathlib import Path
from handData import HandData
from fingerCounter import FingerCounter
from handGeometry import HandGeometry
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MODEL_PATH = (Path(__file__).resolve().parent.parent
    / "Models"
    / "hand_landmarker.task")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontró el modelo de MediaPipe. Ruta esperada: {MODEL_PATH} Asegúrate de tener: Models/hand_landmarker.task")
print("\nCargando modelo de MediaPipe...")
model_buffer = MODEL_PATH.read_bytes()
print(f"Modelo cargado: {len(model_buffer):,} bytes")

#configuracion de mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = (mp.tasks.vision.HandLandmarker)
HandLandmarkerOptions = (mp.tasks.vision.HandLandmarkerOptions)
RunningMode = (mp.tasks.vision.RunningMode)

#configurar handlandmarker
options = HandLandmarkerOptions(
    #Cargar modelo
    base_options=BaseOptions(model_asset_buffer=model_buffer),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7 )

def draw_landmarks(frame,hand_landmarks):
    height, width, _ = frame.shape
    connections = [
        #pulgar
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        #indice
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        #medio
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        #anular
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        #meñique
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        #palma
        (0, 17)
    ]
    for start, end in connections:
        x1 = int(hand_landmarks[start].x* width)
        y1 = int(hand_landmarks[start].y* height)
        x2 = int(hand_landmarks[end].x* width)
        y2 = int(hand_landmarks[end].y* height)
        cv2.line(frame,(x1, y1),(x2, y2),(255, 255, 255),2)
    for index, landmark in enumerate(hand_landmarks):
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame,(x, y),5,(0, 255, 0),-1)
        cv2.putText(frame,str(index),(x + 6, y - 6),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255),1)

camera = cv2.VideoCapture(CAMERA_INDEX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)
if not camera.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

timestamp_ms = 0
previous_time = time.perf_counter()
fps = 0.0
finger_counter = FingerCounter()
hand_geometry = HandGeometry()

print("Hand Landmarker listo.")
print("Iniciando cámara...")
print("\nPresiona Q para salir.\n")
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        success, frame = camera.read()
        if not success:
            print("No se pudo obtener \nel frame de la cámara.")
            break
        #esto hace que sea como un espejo
        frame = cv2.flip(frame,1)
        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)#se le da color a la imagen
        mp_image = mp.Image(image_format=(mp.ImageFormat.SRGB),data=rgb_frame)
        timestamp_ms += 33
        result = (landmarker.detect_for_video(mp_image,timestamp_ms))
        if result.hand_landmarks:
            for (hand_index,hand_landmarks) in enumerate(result.hand_landmarks):
                handedness = (result.handedness[hand_index])
                if not handedness:
                    continue
                hand_label = (handedness[0].category_name)
                confidence = (handedness[0].score)
                if hand_label == "Left":
                    hand_label = "Right"
                elif hand_label == "Right":
                    hand_label = "Left"
                hand = HandData(label=hand_label,confidence=confidence,landmarks=hand_landmarks)
                finger_state = finger_counter.analyze(hand.landmarks)
                hand.fingers = finger_state
                hand.update_finger_count()
                hand.angle = hand_geometry.musical_angle(hand.landmarks)
                finger_pattern = finger_counter.pattern(hand.fingers)
                draw_landmarks(frame,hand.landmarks)
                wrist = (hand.landmarks[0])
                x = int(wrist.x * frame.shape[1])
                y = int(wrist.y * frame.shape[0])
                cv2.putText(frame,f"{hand.label} {hand.confidence:.2f}",(x, y - 20),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 255),2)
                cv2.putText(frame,f"Fingers: {hand.finger_count}",(x, y + 10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255, 255, 255),1)
                cv2.putText(frame,f"Pattern: {finger_pattern}",(x, y + 54),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255, 255, 255),1)
                cv2.putText(frame,f"Gesture: {hand.gesture}",(x, y + 32),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255, 255, 255),1)
                cv2.putText(frame,f"Angle: {hand.angle:.1f} deg",(x, y + 76),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255, 255, 255),1)
                print(hand)
        current_time = (time.perf_counter())
        delta = (current_time - previous_time)
        if delta > 0:
            fps = 1 / delta
        previous_time = current_time
        cv2.putText(frame,f"FPS: {fps:.1f}",(20, 40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0, 255, 0),2)
        cv2.imshow("MusicHands - Vision",frame)
        key = (cv2.waitKey(1)& 0xFF)
        if key == ord("q"):
            break

camera.release()
cv2.destroyAllWindows()
print("\nMusicHands finalizado.")