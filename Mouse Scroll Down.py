import cv2
import mediapipe as mp
import pyautogui

# ── Safety ───────────────────────────────────────────────────────
pyautogui.FAILSAFE = True   # corner of screen → emergency stop
pyautogui.PAUSE = 0

# ── Scroll settings ──────────────────────────────────────────────
SCROLL_AMOUNT   = 20      # lines per scroll "tick"
SCROLL_COOLDOWN = 0      # frames to wait between scrolls (adjust for speed)

# ── MediaPipe setup ──────────────────────────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands

cooldown_counter = 0

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:

    cap = cv2.VideoCapture(0)
    print("Scroll down control started!")
    print("  Raise INDEX finger → scroll DOWN")
    print("  Press 'q' to quit")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        scrolling = False   # is the index finger raised?

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                lm = hand_landmarks.landmark
                index_tip = lm[8]
                index_pip = lm[6]   # middle joint of index finger

                # Index finger raised if tip is higher than pip joint
                if index_tip.y < index_pip.y:
                    scrolling = True

                # ── Visual feedback ──────────────────────────────
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                pip_x, pip_y = int(index_pip.x * w), int(index_pip.y * h)

                # Colour: blue when scrolling, white otherwise
                color = (0, 80, 255) if scrolling else (255, 255, 255)
                cv2.circle(image, (ix, iy), 12, color, -1)
                cv2.putText(image, "INDEX", (ix+15, iy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                # Status text
                if scrolling:
                    status = "SCROLLING DOWN"
                    status_color = (0, 80, 255)
                else:
                    status = "no scroll"
                    status_color = (100, 100, 100)

                cv2.putText(image, status, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        else:
            cv2.putText(image, "No hand detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        # ── Execute scroll with cooldown ────────────────────────
        if cooldown_counter > 0:
            cooldown_counter -= 1

        if scrolling and cooldown_counter == 0:
            pyautogui.scroll(-SCROLL_AMOUNT)   # negative = scroll down
            cooldown_counter = SCROLL_COOLDOWN

        cv2.imshow('Virtual Scroll Down Only', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")
