import cv2
import mediapipe as mp
import pyautogui
import math
 
# ── Safety ───────────────────────────────────────────────────────────────────
pyautogui.FAILSAFE = True   # move mouse to top-left corner to emergency stop
pyautogui.PAUSE = 0         # no delay for faster response
 
# ── Screen size ───────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = pyautogui.size()
 
# ── Tuning ────────────────────────────────────────────────────────────────────
SMOOTHING  = 5    # 1-10: higher = smoother but more laggy
CLICK_DIST = 35   # pinch distance in pixels to trigger a click
 
# ── Setup (same as your working code) ────────────────────────────────────────
mp_drawing = mp.solutions.drawing_utils
mp_hands   = mp.solutions.hands
 
# ── Helper: pixel distance between two landmarks ──────────────────────────────
def distance(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x2 - x1, y2 - y1)
 
# ── Smoothing state ───────────────────────────────────────────────────────────
smooth_x, smooth_y = SCREEN_W // 2, SCREEN_H // 2
clicking      = False
prev_clicking = False
 
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
 
    cap = cv2.VideoCapture(0)
    print("Virtual mouse started!")
    print("  Index finger = move cursor")
    print("  Pinch thumb + index = click")
    print("  Press 'q' to quit")
 
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break
 
        image = cv2.flip(image, 1)
        h, w = image.shape[:2]
 
        # Same BGR→RGB conversion as your working code
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results   = hands.process(image_rgb)
 
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
 
                # Draw skeleton — same as your working code
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
 
                lm = hand_landmarks.landmark
 
                # Landmark 8 = index fingertip → moves cursor
                # Landmark 4 = thumb tip       → pinch to click
                index_tip = lm[8]
                thumb_tip = lm[4]
 
                # ── Move cursor ───────────────────────────────────────────
                # Map index fingertip position to screen coordinates
                target_x = int(index_tip.x * SCREEN_W)
                target_y = int(index_tip.y * SCREEN_H)
 
                # Smooth the movement so cursor isn't jittery
                smooth_x = int(smooth_x + (target_x - smooth_x) / SMOOTHING)
                smooth_y = int(smooth_y + (target_y - smooth_y) / SMOOTHING)
                pyautogui.moveTo(smooth_x, smooth_y)
 
                # ── Detect pinch click ────────────────────────────────────
                pinch_dist = distance(index_tip, thumb_tip, w, h)
                clicking   = pinch_dist < CLICK_DIST
 
                if clicking and not prev_clicking:
                    pyautogui.mouseDown()
                elif not clicking and prev_clicking:
                    pyautogui.mouseUp()
 
                prev_clicking = clicking
 
                # ── Visual feedback on camera window ──────────────────────
                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
 
                dot_color  = (0, 80, 255)  if clicking else (255, 255, 255)
                line_color = (0, 80, 255)  if clicking else (180, 180, 180)
 
                cv2.circle(image, (ix, iy), 12, dot_color, -1)
                cv2.circle(image, (tx, ty), 12, dot_color, -1)
                cv2.line(image, (ix, iy), (tx, ty), line_color, 2)
 
                status       = "CLICKING" if clicking else "moving"
                status_color = (0, 80, 255) if clicking else (100, 220, 100)
                cv2.putText(image, status, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(image, f"pinch: {int(pinch_dist)}px", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            if prev_clicking:
                pyautogui.mouseUp()
                prev_clicking = False
                clicking      = False
            cv2.putText(image, "No hand detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
 
        cv2.imshow('Virtual Mouse', image)
 
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
 
if clicking:
    pyautogui.mouseUp()
 
cap.release()
cv2.destroyAllWindows()
print("Stopped.")
