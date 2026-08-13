import cv2
import mediapipe as mp
CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

camera = cv2.VideoCapture(CAMERA_INDEX)

camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not camera.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")
with mp_hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.7,min_tracking_confidence=0.7) as hands:

    while True:
        success, frame = camera.read()

        if not success:
            print("No se pudo obtener un frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS,mp_drawing_styles.get_default_hand_landmarks_style(),mp_drawing_styles.get_default_hand_connections_style())
        cv2.imshow("MusicHands - Vision",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): #con q salimos
            break
camera.release()
cv2.destroyAllWindows()