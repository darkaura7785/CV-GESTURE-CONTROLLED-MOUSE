import cv2
import mediapipe as mp
import pyautogui
import time
import math

# ========== INITIALIZATION ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

screen_width, screen_height = pyautogui.size()
camera = cv2.VideoCapture(0)

control_enabled = True
last_click_time = 0

# ========== HELPER FUNCTION ==========
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ========== MAIN LOOP ==========
while True:
    success, image = camera.read()
    if not success:
        continue

    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark
            points = {}

            for id in [4, 8, 12, 16, 20]:
                x = int(lm[id].x * w)
                y = int(lm[id].y * h)
                points[id] = (x, y)
                cv2.circle(image, (x, y), 8, (0, 255, 255), -1)

            # ========= START / STOP =========
            if lm[8].y > lm[6].y and lm[12].y > lm[10].y:
                control_enabled = False
                cv2.putText(image, "PAUSED", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                control_enabled = True

            if control_enabled:
                # ========= MOVE MOUSE =========
                ix, iy = points[8]
                mouse_x = int(screen_width / w * ix)
                mouse_y = int(screen_height / h * iy)
                pyautogui.moveTo(mouse_x, mouse_y)

                # ========= LEFT CLICK =========
                if distance(points[8], points[4]) < 40:
                    if time.time() - last_click_time > 0.8:
                        pyautogui.click()
                        last_click_time = time.time()

                # ========= RIGHT CLICK =========
                if distance(points[8], points[12]) < 40:
                    if time.time() - last_click_time > 0.8:
                        pyautogui.rightClick()
                        last_click_time = time.time()

                # ========= SCROLL =========
                if lm[8].y < lm[6].y and lm[12].y < lm[10].y:
                    pyautogui.scroll(40)     # scroll up
                elif lm[8].y > lm[6].y and lm[12].y > lm[10].y:
                    pyautogui.scroll(-40)    # scroll down

    cv2.imshow("Hand Gesture Mouse Controller", image)

    if cv2.waitKey(1) == 27:  # ESC key
        break

camera.release()
cv2.destroyAllWindows()
