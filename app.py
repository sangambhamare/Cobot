import tensorflow as tf
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Load the TFLite model using TensorFlow
def load_model(tflite_model_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
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

# Combine nearby intervals
def combine_nearby_intervals(timing_df, merge_threshold=2.0):
    """
    Combine nearby time intervals in the timing DataFrame.
    
    :param timing_df: DataFrame containing 'Start Time (s)' and 'Stop Time (s)'.
    :param merge_threshold: Maximum difference between stop time of one interval
                            and start time of the next to combine them.
    :return: Merged DataFrame.
    """
    combined_intervals = []
    current_start = timing_df.iloc[0]['Start Time (s)']
    current_stop = timing_df.iloc[0]['Stop Time (s)']

    for i in range(1, len(timing_df)):
        next_start = timing_df.iloc[i]['Start Time (s)']
        next_stop = timing_df.iloc[i]['Stop Time (s)']

        # Check if intervals are close enough to merge
        if next_start - current_stop <= merge_threshold:
            current_stop = next_stop  # Extend the current interval
        else:
            # Save the current interval and start a new one
            combined_intervals.append({"Start Time (s)": current_start, "Stop Time (s)": current_stop})
            current_start = next_start
            current_stop = next_stop

    # Add the last interval
    combined_intervals.append({"Start Time (s)": current_start, "Stop Time (s)": current_stop})

    # Convert to DataFrame
    merged_df = pd.DataFrame(combined_intervals)
    return merged_df

# Streamlit App
st.title("Cobot Activity Detection")
st.write("Upload a video to detect cobot's working and off timings.")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name
        st.success("Video uploaded successfully!")

    # Load TFLite model
    tflite_model_path = "robot_classification_model.tflite"  # Replace with your model's path
    interpreter = load_model(tflite_model_path)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Process video and detect timings
    st.info("Processing video, please wait...")
    timing_df = detect_start_stop_timings(video_path, interpreter, input_details, output_details)

    # Combine nearby intervals
    if not timing_df.empty:
        merged_timing_df = combine_nearby_intervals(timing_df, merge_threshold=2.0)

        # Display timings
        st.write("Cobot Start and Stop Timings:")
        st.dataframe(merged_timing_df)

    else:
        st.warning("No cobot activity detected.")
