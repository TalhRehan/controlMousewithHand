import cv2
import mediapipe as mp
import pyautogui

# Disable the PyAutoGUI fail-safe (use with caution)
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
camera.set(cv2.CAP_PROP_FPS, 30)

x1, y1, x2, y2 = 0, 0, 0, 0
prev_mouse_x, prev_mouse_y = 0, 0

cv2.namedWindow("Hand movement video capture", cv2.WND_PROP_FULLSCREEN)
cv2.resizeWindow("Hand movement video capture", 1280, 720)

while True:
    _, image = camera.read()
    image_height, image_width, _ = image.shape
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    out_hands = hands.process(rgb_image)
    all_hands = out_hands.multi_hand_landmarks

    if all_hands:
        for hand in all_hands:
            mp_draw.draw_landmarks(image, hand)
            one_hand_landmarks = hand.landmark
            for id, lm in enumerate(one_hand_landmarks):
                x = int(lm.x * image_width)
                y = int(lm.y * image_height)

                # Index finger tip (mouse movement)
                if id == 8:
                    mouse_x = int(screen_width / image_width * x * 1.5)
                    mouse_y = int(screen_height / image_height * y * 1.5)
                    mouse_x = (mouse_x + prev_mouse_x * 2) / 3
                    mouse_y = (mouse_y + prev_mouse_y * 2) / 3
                    prev_mouse_x, prev_mouse_y = mouse_x, mouse_y
                    pyautogui.moveTo(mouse_x, mouse_y)
                    cv2.circle(image, (x, y), 10, (0, 255, 255))
                    x1, y1 = x, y

                # Thumb tip (click detection)
                if id == 4:
                    x2, y2 = x, y
                    cv2.circle(image, (x, y), 10, (0, 255, 255))
                    dist = abs(y2 - y1)
                    print(dist)
                    if dist < 40:
                        pyautogui.click()

    cv2.imshow("Hand movement video capture", image)
    key = cv2.waitKey(100)
    if key == 27:
        break

camera.release()
cv2.destroyAllWindows()
