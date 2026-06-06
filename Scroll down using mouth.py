import cv2
import mediapipe as mp
import pyautogui
import math

# ── Safety ───────────────────────────────────────────────────────
pyautogui.FAILSAFE = True   # move mouse to corner → emergency stop
pyautogui.PAUSE = 0

# ── Scroll settings ──────────────────────────────────────────────
SCROLL_AMOUNT          = 15       # lines per scroll tick
SCROLL_COOLDOWN        = 0       # frames to wait between scrolls
MOUTH_OPEN_THRESHOLD   = 0.2     # mouth aspect ratio to consider "open"

# ── MediaPipe Face Mesh ──────────────────────────────────────────
mp_face_mesh   = mp.solutions.face_mesh
mp_drawing     = mp.solutions.drawing_utils
drawing_styles = mp.solutions.drawing_styles

cooldown_counter = 0

with mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,    # set True for more accurate lip landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    cap = cv2.VideoCapture(0)
    print("Mouth scroll started!")
    print("  Open your mouth → scroll DOWN")
    print("  Press 'q' to quit")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        h, w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        mouth_open = False

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Optionally draw full face mesh (can be slow, comment out if needed)
                # mp_drawing.draw_landmarks(
                #     image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style())

                # Key mouth landmarks
                upper_lip = face_landmarks.landmark[13]   # top of upper lip
                lower_lip = face_landmarks.landmark[14]   # bottom of lower lip
                left_corner = face_landmarks.landmark[61]  # left corner
                right_corner = face_landmarks.landmark[291] # right corner

                # Convert to pixel coordinates
                upper = (int(upper_lip.x * w), int(upper_lip.y * h))
                lower = (int(lower_lip.x * w), int(lower_lip.y * h))
                left  = (int(left_corner.x * w), int(left_corner.y * h))
                right = (int(right_corner.x * w), int(right_corner.y * h))

                # Mouth openness: ratio of vertical to horizontal
                vertical_dist   = math.hypot(upper[0]-lower[0], upper[1]-lower[1])
                horizontal_dist = math.hypot(left[0]-right[0], left[1]-right[1])
                if horizontal_dist > 0:
                    ratio = vertical_dist / horizontal_dist
                else:
                    ratio = 0

                mouth_open = ratio > MOUTH_OPEN_THRESHOLD

                # ── Visual feedback ──────────────────────────────
                # Draw mouth bounding points
                color = (0, 80, 255) if mouth_open else (255, 255, 255)
                cv2.circle(image, upper, 5, color, -1)
                cv2.circle(image, lower, 5, color, -1)
                cv2.circle(image, left, 5, color, -1)
                cv2.circle(image, right, 5, color, -1)
                cv2.line(image, upper, lower, color, 2)
                cv2.line(image, left, right, color, 2)

                # Status text
                if mouth_open:
                    status = "MOUTH OPEN - SCROLLING"
                else:
                    status = "mouth closed"
                cv2.putText(image, status, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(image, f"Ratio: {ratio:.2f}", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        else:
            cv2.putText(image, "No face detected", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        # ── Perform scroll with cooldown ────────────────────────
        if cooldown_counter > 0:
            cooldown_counter -= 1

        if mouth_open and cooldown_counter == 0:
            pyautogui.scroll(-SCROLL_AMOUNT)   # negative = scroll down
            cooldown_counter = SCROLL_COOLDOWN

        cv2.imshow('Mouth Scroll', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("Stopped.")
