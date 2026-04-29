import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import tempfile

# Page configuration
st.set_page_config(page_title="Number Plate Detection", layout="wide")

st.title("🚗 Automatic Number Plate Recognition (ANPR)")
st.write("Upload an image or video to detect number plates and extract text.")

# Load model
model = YOLO("best.pt")

# OCR reader
reader = easyocr.Reader(['en'])

uploaded_file = st.file_uploader(
    "Upload Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

detected_numbers = []

# ---------- IMAGE PROCESSING ----------
def process_image(frame):

    results = model(frame)

    for r in results:

        boxes = r.boxes.xyxy.cpu().numpy()

        for box in boxes:

            x1, y1, x2, y2 = map(int, box)

            plate = frame[y1:y2, x1:x2]

            ocr = reader.readtext(plate)

            if len(ocr) > 0:

                text = ocr[0][1]

                detected_numbers.append(text)

                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0,255,0), 2)

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    return frame


# ---------- VIDEO PROCESSING ----------
def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    frame_placeholder = st.empty()

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        for r in results:

            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                plate = frame[y1:y2, x1:x2]

                ocr = reader.readtext(plate)

                if len(ocr) > 0:

                    text = ocr[0][1]

                    detected_numbers.append(text)

                    cv2.putText(frame, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0,255,0), 2)

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        frame_placeholder.image(frame, channels="BGR")

    cap.release()


# ---------- MAIN ----------
if uploaded_file is not None:

    file_type = uploaded_file.type

    # IMAGE
    if "image" in file_type:

        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        frame = cv2.imdecode(file_bytes, 1)

        output = process_image(frame)

        st.image(output, channels="BGR", caption="Detected Plates")


    # VIDEO
    else:

        temp_file = tempfile.NamedTemporaryFile(delete=False)

        temp_file.write(uploaded_file.read())

        st.write("Processing video...")

        process_video(temp_file.name)


    # ---------- RESULTS ----------
    st.subheader("Detected Plate Numbers")

    unique_numbers = list(set(detected_numbers))

    if len(unique_numbers) == 0:

        st.warning("No plate numbers detected")

    else:

        for num in unique_numbers:

            st.success(num)