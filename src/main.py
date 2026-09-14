import cv2
import mediapipe as mp
import numpy as np
from data_logger import save_session

# ==========================================
# MediaPipe setup
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ==========================================
# Angle calculation
# ==========================================

def calculate_angle(a, b, c):

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc)
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    return np.degrees(
        np.arccos(cosine_angle)
    )


# ==========================================
# Exercise settings
# ==========================================

FLEXED_THRESHOLD = 100
EXTENDED_THRESHOLD = 150

# Physiotherapist-defined target
TARGET_EXTENSION = 150


# ==========================================
# Exercise variables
# ==========================================

state = "UNKNOWN"

repetitions = 0
correct_repetitions = 0
incorrect_repetitions = 0

# Used to determine whether current repetition
# reached the target
target_reached = False
min_angle = None
max_angle = None

# ==========================================
# Webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


# ==========================================
# Main loop
# ==========================================

while True:

    success, frame = cap.read()

    if not success:

        print("ERROR: Could not read camera.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)


    # ======================================
    # Hand detected
    # ======================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]


        # ----------------------------------
        # Index finger landmarks
        # ----------------------------------

        mcp = hand_landmarks.landmark[5]
        pip = hand_landmarks.landmark[6]
        dip = hand_landmarks.landmark[7]

        mcp_point = [mcp.x, mcp.y]
        pip_point = [pip.x, pip.y]
        dip_point = [dip.x, dip.y]


        # ----------------------------------
        # Calculate angle
        # ----------------------------------

        angle = calculate_angle(
            mcp_point,
            pip_point,
            dip_point
        )
        if min_angle is None or angle < min_angle:
            min_angle = angle

        if max_angle is None or angle > max_angle:
            max_angle = angle


        # ==================================
        # Target detection
        # ==================================

        if angle >= TARGET_EXTENSION:

            target_reached = True


        # ==================================
        # State machine
        # ==================================

        if angle < FLEXED_THRESHOLD:

            # A complete repetition ends here
            if state == "RETURNING":

                repetitions += 1

                if target_reached:

                    correct_repetitions += 1

                else:

                    incorrect_repetitions += 1

                # Reset for next repetition
                target_reached = False


            state = "FLEXED"


        elif angle >= EXTENDED_THRESHOLD:

            if state == "FLEXED":

                state = "EXTENDED"

            elif state == "RETURNING":

                state = "EXTENDED"


        else:

            if state == "EXTENDED":

                state = "RETURNING"


        # ==================================
        # Draw landmarks
        # ==================================

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )


        # ==================================
        # Current angle
        # ==================================

        cv2.putText(
            frame,
            f"PIP Angle: {angle:.1f} deg",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    # ======================================
    # Display target
    # ======================================

    cv2.putText(
        frame,
        f"Target: {TARGET_EXTENSION} deg",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # ======================================
    # Display state
    # ======================================

    cv2.putText(
        frame,
        f"State: {state}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )


    # ======================================
    # Repetitions
    # ======================================

    cv2.putText(
        frame,
        f"Repetitions: {repetitions}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )


    # ======================================
    # Correct repetitions
    # ======================================

    cv2.putText(
        frame,
        f"Correct: {correct_repetitions}",
        (20, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # ======================================
    # Incorrect repetitions
    # ======================================

    cv2.putText(
        frame,
        f"Incorrect: {incorrect_repetitions}",
        (20, 230),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    # ======================================
    # Accuracy
    # ======================================

    if repetitions > 0:

        accuracy = (
            correct_repetitions /
            repetitions
        ) * 100

    else:

        accuracy = 0


    cv2.putText(
        frame,
        f"Accuracy: {accuracy:.1f}%",
        (20, 270),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================
    # Instructions
    # ======================================

    cv2.putText(
        frame,
        "R = Reset | Q = Quit | S = Save Session",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ======================================
    # Display
    # ======================================

    cv2.imshow(
        "AI Hand Rehabilitation - Exercise",
        frame
    )


    # ======================================
    # Keyboard
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("r"):

        state = "UNKNOWN"

        repetitions = 0
        correct_repetitions = 0
        incorrect_repetitions = 0

        target_reached = False

        print("Exercise data reset.")
    elif key == ord("s"):

        if min_angle is not None and max_angle is not None:
            save_session(
                        max_rom=max_angle,
                        min_rom=min_angle,
                        repetitions=repetitions,
                        correct_repetitions=correct_repetitions,
                        incorrect_repetitions=incorrect_repetitions
                    )
        else:
                print("No ROM data available.")

        
    elif key == ord("q"):

        break


# ==========================================
# Cleanup
# ==========================================

cap.release()
cv2.destroyAllWindows()
hands.close()