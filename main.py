import cv2
import face_recognition
import os
import numpy as np
import threading
from deepface import DeepFace

# Load the pictures of people we know from the folder
known_face_encodings = []
known_face_names = []

for image_name in os.listdir("faces"):
    if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):
        continue
    image_path = os.path.join("faces", image_name)
    try:
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(image_name)[0])
    except Exception as e:
        print(f"Skipping {image_name}: {e}")

# Turn on the webcam (0 is usually the default camera). We use cv2.CAP_DSHOW on Windows to fix some camera errors.
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
video_capture.set(cv2.CAP_PROP_FPS, 60)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_skip = 1
face_locations = []
face_encodings = []
face_names = []
processing = False
scaling_factor = 0.6

def estimate_distance(face_width_pixels):
    known_width_cm = 14.0
    focal_length = 600
    return round((known_width_cm * focal_length) / face_width_pixels, 2)

def name_to_color(name):
    np.random.seed(hash(name) % 2**32)
    return tuple(np.random.randint(100, 256, 3).tolist())

# A little math to fix the age guess so it's not jumping around
def estimate_real_age(estimated_age):
    x = estimated_age
    linear_part = 1.92 * x - 26.5
    quadratic_part = 0.05 * (x - 27)**2
    return max(0, linear_part + quadratic_part)

# Fix the lighting if the room is too dark or bright
def adaptive_gamma_correction(image, target_mean=128):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean = gray.mean()
    if mean == 0:
        gamma = 1.0
    else:
        gamma = np.log(target_mean / 255) / np.log(mean / 255)
    inv_gamma = 1.0 / gamma if gamma != 0 else 1.0

    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# Make the image look sharper and clearer
def clahe_enhancement(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def detect_attributes(face_image):
    try:
# Make the face image better so the AI can read it easily
        face_image = adaptive_gamma_correction(face_image)
        face_image = clahe_enhancement(face_image)

        result = DeepFace.analyze(
            face_image,
            actions=['age', 'gender', 'emotion'],
            enforce_detection=False,
            detector_backend='opencv',
        )[0]

        gender = f"Est. Gender: {result['dominant_gender'].capitalize()}"
        age = f"Est. Age: {round(estimate_real_age(result['age']))} yrs"
        mood = f"Est. Mood: {result['dominant_emotion'].capitalize()}"
        return gender, age, mood
    except Exception as e:
        print(f"[DeepFace Error] {e}")
        return "Est. Gender: Unknown", "Est. Age: Unknown", "Est. Mood: Unknown"

def process_frame(frame):
    global face_locations, face_encodings, face_names, processing
    processing = True

    small_frame = cv2.resize(frame, (0, 0), fx=scaling_factor, fy=scaling_factor)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    # Try to find faces, but don't crash if something goes wrong with the image type
    try:
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    except Exception as e:
        print(f"Error in face recognition: {e}")
        processing = False
        return

    new_face_names = []
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.35)
        name = "Unknown"
        color = (0, 0, 255)

        if True in matches:
            matched_idxs = [i for i, match in enumerate(matches) if match]
            name = known_face_names[matched_idxs[0]]
            color = name_to_color(name)

        top, right, bottom, left = face_location
        face_image = rgb_frame[top:bottom, left:right]
        face_width = right - left
        distance = f"Est. Distance: {estimate_distance(face_width)} cm"

        if face_image.size != 0:
            gender, age, mood = detect_attributes(face_image)
            details = [f"Name: {name}", gender, age, mood, distance]
            name = "\n".join(details)

        new_face_names.append((name, color))

    face_names = new_face_names
    processing = False

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    if not processing:
        threading.Thread(target=process_frame, args=(frame.copy(),), daemon=True).start()

    for (top, right, bottom, left), (name, color) in zip(face_locations, face_names):
        top, right, bottom, left = int(top / scaling_factor), int(right / scaling_factor), int(bottom / scaling_factor), int(left / scaling_factor)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        y_offset = top
        for line in name.split("\n"):
            cv2.putText(frame, line, (right + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (right + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            y_offset += 30

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()