import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
from pathlib import Path

# Set up the app
st.title("Cobot Activity Detection")
st.write("Upload a video or record in real-time to detect when the cobot is working.")

# Load TFLite model
def load_model(tflite_model_path):
    interpreter = tflite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

# Process video frames with the TFLite model
def process_frame(frame, interpreter, input_details, output_details):
    frame_resized = cv2.resize(frame, (224, 224))
    frame_normalized = np.expand_dims(frame_resized / 255.0, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], frame_normalized)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return prediction > 0.5  # Return True for working, False for off

# Detect start and stop timings
def detect_start_stop_timings(video_path, interpreter, input_details, output_details, threshold=0.5, fps=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = None
    stop_time = None
    timings = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through the model
        is_working = process_frame(frame, interpreter, input_details, output_details)

        if is_working:
            if start_time is None:
                start_time = frame_count / fps
        else:
            if start_time is not None:
                stop_time = (frame_count - 1) / fps
                timings.append({"Start Time (s)": start_time, "Stop Time (s)": stop_time})
                start_time = None  # Reset start time

        frame_count += 1

    # Handle if the video ends while working
    if start_time is not None:
        timings.append({"Start Time (s)": start_time, "Stop Time (s)": frame_count / fps})

    cap.release()
    timing_df = pd.DataFrame(timings)
    return timing_df

# Handle video upload
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
real_time_video = st.button("Record Real-Time Video")

# Temporary file for uploaded video
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name
        st.success("Video uploaded successfully!")

# Handle real-time video
if real_time_video:
    video_path = "real_time_video.mp4"  # Placeholder for real-time video implementation
    st.warning("Real-time video recording feature not implemented yet.")

# Process the video if we have a valid path
if 'video_path' in locals():
    tflite_model_path = "robot_classification_model.tflite"  # Path to your TFLite model
    interpreter = load_model(tflite_model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    st.info("Processing video, please wait...")
    timing_df = detect_start_stop_timings(video_path, interpreter, input_details, output_details)

    # Display the table
    if not timing_df.empty:
        st.write("Cobot Start and Stop Timings:")
        st.dataframe(timing_df)
    else:
        st.warning("No cobot activity detected in the video.")
