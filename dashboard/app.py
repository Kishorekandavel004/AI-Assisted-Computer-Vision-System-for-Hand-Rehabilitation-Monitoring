import os
import sys
import cv2
import av
import pickle
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Hand Rehabilitation",
    page_icon="🖐️",
    layout="wide"
)


# =========================================================
# IMPORT DATA LOGGER
# =========================================================

from src.data_logger import save_session


# =========================================================
# FILE PATHS
# =========================================================

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "session_data.csv"
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "models",
    "rom_prediction_model.pkl"
)


# =========================================================
# LOAD ML MODEL
# =========================================================

ml_model = None
ml_features = None
ml_model_name = None

if os.path.exists(MODEL_FILE):

    try:

        with open(MODEL_FILE, "rb") as file:

            model_data = pickle.load(file)

        ml_model = model_data["model"]
        ml_features = model_data["features"]
        ml_model_name = model_data["model_name"]

    except Exception as e:

        ml_model = None

        st.warning(
            f"ML model could not be loaded: {e}"
        )


# =========================================================
# TITLE
# =========================================================

st.markdown(
    """
    <h1 style="text-align:center;">
    🖐️ AI-Assisted Hand Rehabilitation System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center;">
    Computer Vision Based Index Finger Rehabilitation Monitoring
    </p>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

defaults = {

    "session_active": False,

    "session_finished": False,

    "session_saved": False,

    "patient_id": "P001",

    "exercise": "Index Finger Flexion-Extension",

    "target_extension": 150,

    "target_repetitions": 10,

    "current_angle": 0.0,

    "min_angle": 0.0,

    "max_angle": 0.0,

    "rom": 0.0,

    "repetitions": 0,

    "correct_repetitions": 0,

    "incorrect_repetitions": 0,

    "accuracy": 0.0
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙️ Session Settings")

patient_id = st.sidebar.text_input(
    "Patient ID",
    value=st.session_state.patient_id
)

exercise = st.sidebar.selectbox(
    "Exercise",
    [
        "Index Finger Flexion-Extension"
    ]
)

target_extension = st.sidebar.slider(
    "Target Extension Angle",
    min_value=90,
    max_value=180,
    value=int(
        st.session_state.target_extension
    ),
    step=1
)

target_repetitions = st.sidebar.number_input(
    "Target Repetitions",
    min_value=1,
    max_value=100,
    value=int(
        st.session_state.target_repetitions
    ),
    step=1
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Patient Instructions**

    1. Place your hand clearly in front of the camera.
    2. Keep the index finger visible.
    3. Start the session.
    4. Slowly flex and extend the index finger.
    5. Return to the starting position after each repetition.
    6. Finish and save the session.
    """
)


# =========================================================
# ANGLE CALCULATION
# =========================================================

def calculate_angle(a, b, c):

    a = np.array(a)

    b = np.array(b)

    c = np.array(c)

    ba = a - b

    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator == 0:

        return 0.0

    cosine_angle = (
        np.dot(ba, bc) /
        denominator
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return float(angle)


# =========================================================
# MEDIAPIPE
# =========================================================

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils


# =========================================================
# HAND PROCESSOR
# =========================================================

class HandProcessor(VideoProcessorBase):

    def __init__(self, target_extension=150):

        self.target_extension = target_extension

        self.hands = mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=1,

            min_detection_confidence=0.6,

            min_tracking_confidence=0.6
        )

        # ---------------------------------------------
        # Angle
        # ---------------------------------------------

        self.current_angle = 0.0

        self.min_angle = None

        self.max_angle = None

        # ---------------------------------------------
        # Repetitions
        # ---------------------------------------------

        self.repetitions = 0

        self.correct_repetitions = 0

        self.incorrect_repetitions = 0

        # ---------------------------------------------
        # Movement state
        # ---------------------------------------------

        self.state = "UNKNOWN"

        self.target_reached = False

    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        # Mirror webcam
        image = cv2.flip(
            image,
            1
        )

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(
            rgb_image
        )

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            # -----------------------------------------
            # INDEX FINGER LANDMARKS
            #
            # MCP = 5
            # PIP = 6
            # DIP = 7
            # -----------------------------------------

            mcp = hand_landmarks.landmark[5]

            pip = hand_landmarks.landmark[6]

            dip = hand_landmarks.landmark[7]

            angle = calculate_angle(

                [mcp.x, mcp.y],

                [pip.x, pip.y],

                [dip.x, dip.y]
            )

            self.current_angle = angle

            # -----------------------------------------
            # MINIMUM ANGLE
            # -----------------------------------------

            if self.min_angle is None:

                self.min_angle = angle

            else:

                self.min_angle = min(
                    self.min_angle,
                    angle
                )

            # -----------------------------------------
            # MAXIMUM ANGLE
            # -----------------------------------------

            if self.max_angle is None:

                self.max_angle = angle

            else:

                self.max_angle = max(
                    self.max_angle,
                    angle
                )

            # -----------------------------------------
            # TARGET REACHED
            # -----------------------------------------

            if angle >= self.target_extension:

                self.target_reached = True

            # -----------------------------------------
            # REPETITION STATE MACHINE
            #
            # FLEXED < 100°
            # MOVING 100° to target
            # EXTENDED >= target
            #
            # A repetition is completed when the
            # finger returns to the flexed position.
            # -----------------------------------------

            if angle < 100:

                # A movement was previously started
                if self.state in [
                    "MOVING",
                    "EXTENDED",
                    "RETURNING"
                ]:

                    self.repetitions += 1

                    if self.target_reached:

                        self.correct_repetitions += 1

                    else:

                        self.incorrect_repetitions += 1

                    self.target_reached = False

                self.state = "FLEXED"

            elif angle >= self.target_extension:

                self.state = "EXTENDED"

            elif angle >= 100:

                if self.state == "FLEXED":

                    self.state = "MOVING"

                elif self.state == "EXTENDED":

                    self.state = "RETURNING"

                elif self.state == "MOVING":

                    self.state = "MOVING"

                elif self.state == "RETURNING":

                    self.state = "RETURNING"

            # -----------------------------------------
            # DRAW LANDMARKS
            # -----------------------------------------

            mp_drawing.draw_landmarks(

                image,

                hand_landmarks,

                mp_hands.HAND_CONNECTIONS
            )

            # -----------------------------------------
            # CAMERA TEXT
            # -----------------------------------------

            cv2.putText(

                image,

                f"Angle: {angle:.1f} deg",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )

            cv2.putText(

                image,

                f"Repetitions: {self.repetitions}",

                (20, 75),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 0),

                2
            )

            cv2.putText(

                image,

                f"Target: {self.target_extension} deg",

                (20, 110),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2
            )

            cv2.putText(

                image,

                f"State: {self.state}",

                (20, 145),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (255, 255, 255),

                2
            )

        else:

            cv2.putText(

                image,

                "Hand not detected",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 0, 255),

                2
            )

        return av.VideoFrame.from_ndarray(

            image,

            format="bgr24"
        )


# =========================================================
# WEBRTC CONFIGURATION
# =========================================================

RTC_CONFIGURATION = RTCConfiguration(

    {
        "iceServers": [

            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }

        ]
    }
)


# =========================================================
# SESSION INFORMATION
# =========================================================

st.markdown("---")

st.subheader("🎯 Rehabilitation Session")

info1, info2, info3 = st.columns(3)

with info1:

    st.write(
        f"**Patient ID:** {patient_id}"
    )

with info2:

    st.write(
        f"**Exercise:** {exercise}"
    )

with info3:

    st.write(
        f"**Target:** {target_extension}° "
        f"| **Repetitions:** {target_repetitions}"
    )


# =========================================================
# START NEW SESSION
# =========================================================

if not st.session_state.session_active:

    if st.button(
        "▶️ Start New Session",
        use_container_width=True
    ):

        # Store current settings
        st.session_state.patient_id = patient_id

        st.session_state.exercise = exercise

        st.session_state.target_extension = (
            target_extension
        )

        st.session_state.target_repetitions = (
            target_repetitions
        )

        # Reset result values
        st.session_state.current_angle = 0.0

        st.session_state.min_angle = 0.0

        st.session_state.max_angle = 0.0

        st.session_state.rom = 0.0

        st.session_state.repetitions = 0

        st.session_state.correct_repetitions = 0

        st.session_state.incorrect_repetitions = 0

        st.session_state.accuracy = 0.0

        st.session_state.session_saved = False

        st.session_state.session_finished = False

        st.session_state.session_active = True

        st.rerun()


# =========================================================
# ACTIVE SESSION
# =========================================================

if st.session_state.session_active:

    st.info(
        "🖐️ Perform index finger flexion-extension. "
        "Return the finger to the flexed position "
        "after extension to complete one repetition."
    )

    webrtc_ctx = webrtc_streamer(

        key="hand-rehabilitation",

        video_processor_factory=lambda:
            HandProcessor(
                target_extension
            ),

        rtc_configuration=RTC_CONFIGURATION,

        media_stream_constraints={

            "video": True,

            "audio": False
        },

        async_processing=True
    )

    # =====================================================
    # LIVE DATA
    # =====================================================

    if webrtc_ctx.video_processor:

        processor = (
            webrtc_ctx.video_processor
        )

        # ---------------------------------------------
        # ROM
        # ---------------------------------------------

        if (
            processor.min_angle is not None
            and
            processor.max_angle is not None
        ):

            live_rom = (
                processor.max_angle
                -
                processor.min_angle
            )

        else:

            live_rom = 0.0

        # ---------------------------------------------
        # ACCURACY
        # ---------------------------------------------

        if processor.repetitions > 0:

            live_accuracy = (

                processor.correct_repetitions
                /
                processor.repetitions

            ) * 100

        else:

            live_accuracy = 0.0

        # ---------------------------------------------
        # LIVE METRICS
        # ---------------------------------------------

        st.markdown("### 📊 Live Session")

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                "Current Angle",
                f"{processor.current_angle:.1f}°"
            )

        with c2:

            st.metric(
                "Repetitions",
                f"{processor.repetitions} / "
                f"{target_repetitions}"
            )

        with c3:

            st.metric(
                "Movement Range",
                f"{live_rom:.1f}°"
            )

        with c4:

            st.metric(
                "Accuracy",
                f"{live_accuracy:.1f}%"
            )

        # ---------------------------------------------
        # REPETITION DETAILS
        # ---------------------------------------------

        st.markdown("### 🔎 Repetition Details")

        r1, r2, r3 = st.columns(3)

        with r1:

            st.metric(
                "Correct",
                processor.correct_repetitions
            )

        with r2:

            st.metric(
                "Incorrect",
                processor.incorrect_repetitions
            )

        with r3:

            st.metric(
                "Target Angle",
                f"{target_extension}°"
            )

        # ---------------------------------------------
        # FINISH SESSION
        # ---------------------------------------------

        st.markdown("---")

        if st.button(
            "⏹️ Finish Session",
            use_container_width=True
        ):

            # -----------------------------------------
            # COPY PROCESSOR VALUES TO SESSION STATE
            # -----------------------------------------

            st.session_state.current_angle = (
                processor.current_angle
            )

            st.session_state.min_angle = (
                processor.min_angle
                if processor.min_angle is not None
                else 0.0
            )

            st.session_state.max_angle = (
                processor.max_angle
                if processor.max_angle is not None
                else 0.0
            )

            st.session_state.rom = live_rom

            st.session_state.repetitions = (
                processor.repetitions
            )

            st.session_state.correct_repetitions = (
                processor.correct_repetitions
            )

            st.session_state.incorrect_repetitions = (
                processor.incorrect_repetitions
            )

            st.session_state.accuracy = (
                live_accuracy
            )

            # -----------------------------------------
            # END SESSION
            # -----------------------------------------

            st.session_state.session_active = False

            st.session_state.session_finished = True

            st.rerun()


# =========================================================
# SESSION RESULT
# =========================================================

if st.session_state.session_finished:

    st.markdown("---")

    st.subheader("📋 Session Result")

    result1, result2, result3, result4 = st.columns(4)

    with result1:

        st.metric(
            "Movement Range",
            f"{st.session_state.rom:.1f}°"
        )

    with result2:

        st.metric(
            "Repetitions",
            st.session_state.repetitions
        )

    with result3:

        st.metric(
            "Correct",
            st.session_state.correct_repetitions
        )

    with result4:

        st.metric(
            "Accuracy",
            f"{st.session_state.accuracy:.1f}%"
        )

    # =====================================================
    # SAVE SESSION
    # =====================================================

    st.markdown("### 💾 Save Session")

    if not st.session_state.session_saved:

        if st.button(
            "💾 Save Session",
            type="primary",
            use_container_width=True
        ):

            try:

                save_session(

                    patient_id=(
                        st.session_state.patient_id
                    ),

                    exercise=(
                        st.session_state.exercise
                    ),

                    target_angle=(
                        st.session_state.target_extension
                    ),

                    max_rom=(
                        st.session_state.max_angle
                    ),

                    min_rom=(
                        st.session_state.min_angle
                    ),

                    repetitions=(
                        st.session_state.repetitions
                    ),

                    correct_repetitions=(
                        st.session_state.correct_repetitions
                    ),

                    incorrect_repetitions=(
                        st.session_state.incorrect_repetitions
                    )
                )

                st.session_state.session_saved = True

                st.success(
                    "✅ Session saved successfully!"
                )

            except Exception as e:

                st.error(
                    f"Unable to save session: {e}"
                )

    else:

        st.success(
            "✅ This session has already been saved."
        )


# =========================================================
# AI PROGRESS PREDICTION
# =========================================================

st.markdown("---")

st.subheader("🤖 AI Progress Prediction")


if (
    ml_model is not None
    and os.path.exists(DATA_FILE)
):

    try:

        prediction_df = pd.read_csv(
            DATA_FILE
        )

        if len(prediction_df) >= 4:

            # -----------------------------------------
            # SORT BY DATE AND TIME
            # -----------------------------------------

            prediction_df["DateTime"] = (
                pd.to_datetime(
                    prediction_df["Date"]
                    .astype(str)
                    + " "
                    +
                    prediction_df["Time"]
                    .astype(str)
                )
            )

            prediction_df = (
                prediction_df
                .sort_values("DateTime")
                .reset_index(drop=True)
            )

            # -----------------------------------------
            # LATEST SESSION
            # -----------------------------------------

            latest = prediction_df.iloc[-1]

            current_rom = float(
                latest["ROM_Range"]
            )

            previous_rom = float(
                prediction_df[
                    "ROM_Range"
                ].iloc[-2]
            )

            rom_change = (
                current_rom -
                previous_rom
            )

            # Previous 3 sessions
            previous_values = (
                prediction_df[
                    "ROM_Range"
                ]
                .iloc[-4:-1]
            )

            previous_average_rom = (
                previous_values.mean()
            )

            # -----------------------------------------
            # CREATE ML INPUT
            # -----------------------------------------

            prediction_input = pd.DataFrame({

                "ROM_Range": [
                    current_rom
                ],

                "Previous_ROM": [
                    previous_rom
                ],

                "ROM_Change": [
                    rom_change
                ],

                "Previous_Average_ROM": [
                    previous_average_rom
                ],

                "Repetitions": [
                    latest["Repetitions"]
                ],

                "Correct_Repetitions": [
                    latest[
                        "Correct_Repetitions"
                    ]
                ],

                "Incorrect_Repetitions": [
                    latest[
                        "Incorrect_Repetitions"
                    ]
                ],

                "Accuracy": [
                    latest["Accuracy"]
                ],

                "Target_Angle": [
                    latest["Target_Angle"]
                ]
            })

            # -----------------------------------------
            # MATCH TRAINING FEATURES
            # -----------------------------------------

            prediction_input = (
                prediction_input[
                    ml_features
                ]
            )

            # -----------------------------------------
            # PREDICTION
            # -----------------------------------------

            predicted_rom = (
                ml_model
                .predict(
                    prediction_input
                )[0]
            )

            predicted_rom = max(
                0.0,
                float(predicted_rom)
            )

            # -----------------------------------------
            # TREND
            # -----------------------------------------

            difference = (
                predicted_rom -
                current_rom
            )

            if difference > 2:

                trend = "↗ Improving"

            elif difference < -2:

                trend = "↘ Decreasing"

            else:

                trend = "→ Stable"

            # -----------------------------------------
            # DISPLAY
            # -----------------------------------------

            p1, p2, p3 = st.columns(3)

            with p1:

                st.metric(
                    "Current ROM",
                    f"{current_rom:.1f}°"
                )

            with p2:

                st.metric(
                    "Predicted Next ROM",
                    f"{predicted_rom:.1f}°"
                )

            with p3:

                st.metric(
                    "Predicted Trend",
                    trend
                )

            st.info(
                "The AI prediction estimates the "
                "movement range of the next session "
                "from previous session measurements."
            )

            st.caption(
                f"Model used: {ml_model_name}"
            )

        else:

            st.info(
                "Complete at least 4 sessions "
                "to generate an AI prediction."
            )

    except Exception as e:

        st.warning(
            f"AI prediction unavailable: {e}"
        )

else:

    st.info(
        "AI prediction will appear after the "
        "trained model is available."
    )


# =========================================================
# PROGRESS
# =========================================================

st.markdown("---")

st.subheader("📈 Progress")


if os.path.exists(DATA_FILE):

    try:

        df = pd.read_csv(
            DATA_FILE
        )

        if len(df) > 0:

            # -----------------------------------------
            # SUMMARY
            # -----------------------------------------

            s1, s2, s3, s4 = st.columns(4)

            with s1:

                st.metric(
                    "Total Sessions",
                    len(df)
                )

            with s2:

                st.metric(
                    "Average ROM",
                    f"{df['ROM_Range'].mean():.1f}°"
                )

            with s3:

                st.metric(
                    "Average Accuracy",
                    f"{df['Accuracy'].mean():.1f}%"
                )

            with s4:

                st.metric(
                    "Best ROM",
                    f"{df['ROM_Range'].max():.1f}°"
                )

            # -----------------------------------------
            # ROM GRAPH
            # -----------------------------------------

            st.markdown("### 📊 ROM Progress")

            chart_df = df.copy()

            chart_df["Session"] = range(
                1,
                len(chart_df) + 1
            )

            st.line_chart(

                chart_df.set_index(
                    "Session"
                )[
                    ["ROM_Range"]
                ]
            )

            # -----------------------------------------
            # ACCURACY GRAPH
            # -----------------------------------------

            st.markdown(
                "### 🎯 Accuracy Progress"
            )

            st.line_chart(

                chart_df.set_index(
                    "Session"
                )[
                    ["Accuracy"]
                ]
            )

            # -----------------------------------------
            # SESSION HISTORY
            # -----------------------------------------

            st.markdown(
                "### 🗂️ Session History"
            )

            display_columns = [
                "Patient_ID",
                "Date",
                "Time",
                "Exercise",
                "Target_Angle",
                "ROM_Range",
                "Repetitions",
                "Correct_Repetitions",
                "Incorrect_Repetitions",
                "Accuracy"
            ]

            available_columns = [
                column
                for column in display_columns
                if column in df.columns
            ]

            st.dataframe(
                df[available_columns],
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No saved sessions yet."
            )

    except Exception as e:

        st.error(
            f"Error reading session data: {e}"
        )

else:

    st.info(
        "No saved sessions yet."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "AI-Assisted Hand Rehabilitation Monitoring System "
    "| MediaPipe + OpenCV + Machine Learning"
)