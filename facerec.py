#!/usr/bin/env python
import sys
import cv2
import numpy as np
from openvino import Core
import os
import subprocess
import pickle
import time
DIR = os.path.dirname(__file__)
CAPTURE_PATH="/dev/video2"
CONFIDENCE_THRESHOLD=0.7
IMPROVE_THRESHOLD=0.8
FIRST_TRAIN_SIZE=25
USAGE = f"""\
Usage:
    {sys.argv[0]} add [face_name]
        - Add a new face for recognition.
        - Optionally specify a face_name (default is next available integer).

    {sys.argv[0]} remove [face_name1 face_name2 ...]
        - Remove one or more saved faces by their names.
        - If 'all' is provided, all saved faces for the current user will be deleted.

    {sys.argv[0]} check
        - Attempt to recognize the current user using the webcam.

Notes:
    - Recognition and training use OpenVINO models located in ./models.
    - sudo privileges are needed to save or delete faces.
"""

# --- Initialize OpenVINO ---
core = Core()

# Load models
# 1. Face Detection Model
det_model_path = os.path.join(DIR,"models/face-detection-retail-0005.xml")
det_model = core.read_model(det_model_path)
compiled_det = core.compile_model(det_model, "CPU")

# 3. Face Recognition Model
rec_model_path = os.path.join(DIR,"models/face-reidentification-retail-0095.xml")
rec_model = core.read_model(rec_model_path)
compiled_rec = core.compile_model(rec_model, "CPU")

def ensure_bgr(frame):
    """Ensure a frame is in BGR 3-channel format."""
    if frame is None:
        raise ValueError("Input frame is None")

    if len(frame.shape) == 2:
        # Grayscale image (height, width)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3:
        if frame.shape[2] == 1:
            # Grayscale with 1-channel
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            # Already BGR
            return frame
        elif frame.shape[2] == 4:
            # BGRA -> BGR (remove alpha channel)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unsupported number of channels: {frame.shape[2]}")
    else:
        raise ValueError("Unsupported frame shape")

# Image improvements
def gray_world_correction(img):
    avg_b = np.mean(img[:, :, 0])
    avg_g = np.mean(img[:, :, 1])
    avg_r = np.mean(img[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    img[:, :, 0] = np.clip(img[:, :, 0] * scale_b, 0, 255)
    img[:, :, 1] = np.clip(img[:, :, 1] * scale_g, 0, 255)
    img[:, :, 2] = np.clip(img[:, :, 2] * scale_r, 0, 255)

    return img.astype(np.uint8)

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced

def is_blurry(image, threshold=100.0):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var < threshold

def check_quality(image):
    to_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.size == 0:
        return False
    if is_blurry(to_check):
        return False
    # avoid light problems
    if to_check.mean() < 75 or to_check.mean() > 180:
        return False
    # better contrast
    if to_check.std() <20:
        return False
    return True
def check_allowed(image): # don't try if laptop camera is obstructed
    to_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return False if to_check.mean()<20 else True

# --- Utility: Cosine Similarity ---
def cosine_similarity(a, b):
    """Check the cosine similarity of two vertex arrays"""
    a = a.flatten()  # Convert to 1D
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

#----This part load the face data-----------
if os.path.exists(os.path.join(DIR,"preload_embeddings.pkl")):
    with open(os.path.join(DIR,"preload_embeddings.pkl"), "rb") as f:
        ref_embeddings = pickle.load(f)
        if type(ref_embeddings)==list: # single-user format
            ref_embeddings={os.environ["USER"]:{0:ref_embeddings}}
else:
    ref_embeddings = {os.environ["USER"]:{}}
for user, faces in enumerate(ref_embeddings): # turn in  dictionnary for easier embedding
    if type(faces)==list:
        ref_embeddings[user]={str(i): content for i, content in enumerate(faces)}
#----End data loader-----------------------

# --- Helper: Process detection results ---
def parse_detections(detections, frame_shape, conf_threshold=0.5):
    h, w = frame_shape[:2]
    boxes = []
    # Assuming the detection model output is in the format:
    # [image_id, label, conf, xmin, ymin, xmax, ymax] for each detection.
    for detection in detections[0][0]:
        confidence = float(detection[2])
        if confidence > conf_threshold:
            xmin = int(detection[3] * w)
            ymin = int(detection[4] * h)
            xmax = int(detection[5] * w)
            ymax = int(detection[6] * h)
            boxes.append((xmin, ymin, xmax, ymax, confidence))
    return boxes

if len(ref_embeddings)>100: # lighten the presaved things
    final_embeddings=[]
    for embed in ref_embeddings:
        # Compare with every reference embedding
        similarities = [cosine_similarity(ref_emb, embed) for ref_emb in final_embeddings]
        # Keep the face only if there is no very similar ones in the base
        is_unsimilar = all([sim < 0.8 for sim in similarities])
        if is_unsimilar or len(final_embeddings)==0:
            final_embeddings.append(embed)
    ref_embeddings=final_embeddings
    # save the lightened preload
    with open(os.path.join(DIR,"preload_embeddings.pkl"), "wb") as f:
        pickle.dump(ref_embeddings, f)

def check(username,n_try=5):
    """
    Deserves to the face recognition in itself.
    Called by the daemon when it receives an auth request
    take as argument the numbers of frames to check, spaced by ~0.1 second
    """
    if not username in ref_embeddings:
        return "fail"
    if len(ref_embeddings[username])==0:
        return "Error: training base is too small"
    cap = cv2.VideoCapture(CAPTURE_PATH, cv2.CAP_V4L2)
    if not cap.isOpened():
        return "Error: Failed to open Webcam"
    attempts_count=0
    
    while attempts_count<n_try:
        did_try=0
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = ensure_bgr(frame)
        if not check_allowed(frame):
            return "fail"
        frame = apply_clahe(frame)
        frame = gray_world_correction(frame)
        frame = cv2.filter2D(frame,-1,np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        # 1. Detect faces using the detection model
        # Preprocess frame for detection (resize to model's expected input, e.g., 300x300)
        det_input_size = (300, 300)  # Adjust if needed
        frame_resized = cv2.resize(frame, det_input_size)
        # Assume detection model requires CHW; adjust conversion if needed
        input_blob_det = frame_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        # Run detection
        det_result = compiled_det([input_blob_det])[compiled_det.output(0)]
        # Parse detection output (update parsing based on your model’s output format)
        boxes = parse_detections(det_result, frame.shape, conf_threshold=0.5)
        for (xmin, ymin, xmax, ymax, conf) in boxes:
            # Crop the detected face from the original frame
            face_crop = frame[ymin:ymax, xmin:xmax]
            if not check_quality(face_crop):
                continue
            did_try=1
            # 3. Run recognition on the face crop
            rec_face = cv2.resize(face_crop, (128, 128))
            rec_face = gray_world_correction(rec_face)
            rec_input = rec_face.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            rec_embedding = compiled_rec([rec_input])[compiled_rec.output(0)]
            # Compare with every reference embedding
            for key, user_face in ref_embeddings[username].items():
                similarities = [cosine_similarity(ref_emb, rec_embedding) for ref_emb in user_face]
                # You can adjust your threshold accordingly
                is_match = any([sim > CONFIDENCE_THRESHOLD for sim in similarities])
                if is_match:
                    cap.release()
                    if all([sim < IMPROVE_THRESHOLD for sim in similarities]) and len(user_face)<500: # use as training image
                        user_face.append(rec_embedding)
                        with open(os.path.join(DIR,"preload_embeddings.pkl"), "wb") as f:
                            pickle.dump(ref_embeddings, f)
                    return "pass"
        attempts_count+=did_try
        time.sleep(0.05)
    cap.release()
    return "fail"
    

    
def add_face(face_name=...):
    """
    python facerec.py add
    Allow to save needed data for good performance on first unlock attempts.
    The recognition will then be trained more with successfull recognitions that are not too much similar to the original one.
    
    !WARN! That can be unsafe, as if your camera takes too bad photos, this can help someone else to get recognized more easely.
    
    But in most cases, it just improve performance (you need less frames to be recognized).
    """
    username=os.environ["USER"]
    new_face=[]
    if not username in ref_embeddings:
        ref_embeddings[username]={}
    elif face_name in ref_embeddings[username]:
        print("Unable to save face with this name, because it has already been taken...")
    if face_name==...: # use the first integer that was not used
        i=0
        while str(i) in ref_embeddings[username]: i+=1
        face_name=str(i)
    ref_embeddings[username][face_name]=new_face
    cap = cv2.VideoCapture("/dev/video2", cv2.CAP_V4L2)
    if not cap.isOpened():
        return "Error: Failed to open Webcam"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = ensure_bgr(frame)
        frame = apply_clahe(frame)
        frame = gray_world_correction(frame)
        frame = cv2.filter2D(frame,-1,np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        # 1. Detect faces using the detection model
        # Preprocess frame for detection (resize to model's expected input, e.g., 300x300)
        det_input_size = (300, 300)  # Adjust if needed
        frame_resized = cv2.resize(frame, det_input_size)
        # Assume detection model requires CHW; adjust conversion if needed
        input_blob_det = frame_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        # Run detection
        det_result = compiled_det([input_blob_det])[compiled_det.output(0)]
        # Parse detection output (update parsing based on your model’s output format)
        boxes = parse_detections(det_result, frame.shape, conf_threshold=0.5)
        for (xmin, ymin, xmax, ymax, conf) in boxes:
            # Draw box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            # Crop the detected face from the original frame
            face_crop = frame[ymin:ymax, xmin:xmax]
            if not check_quality(face_crop):
                continue
            # 3. Run recognition on the face crop
            rec_face = cv2.resize(face_crop, (128, 128))
            rec_face = gray_world_correction(rec_face)
            rec_input = rec_face.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            rec_embedding = compiled_rec([rec_input])[compiled_rec.output(0)]
            # Compare with every reference embedding
            similarities = [cosine_similarity(ref_emb, rec_embedding) for ref_emb in new_face]
            face_preview = cv2.resize(rec_face, (128, 128))
            cv2.imshow("Face Crop", rec_face)
            # You can adjust your threshold accordingly
            if (all([0.4 < sim < IMPROVE_THRESHOLD for sim in similarities]) and len(new_face)<FIRST_TRAIN_SIZE) or len(new_face)==0: # use as training image
                new_face.append(rec_embedding)
                print("",len(new_face),"/",FIRST_TRAIN_SIZE,"training frames saved...", end="\r")
        cv2.imshow("Webcam - Press 's' to save face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(new_face)>=FIRST_TRAIN_SIZE:
            print("Succeeded to record needed data...\nYour face has been recorded.")
            break
                
    cap.release()
    # save faces as vertex data (safer than images and faster to load)
    tmp_path="/tmp/facerec.tmp"
    print("Please enter your password in order to save your new face:")
    with open(tmp_path, "wb") as f:
        pickle.dump(ref_embeddings, f)
    try:
        subprocess.check_output(["sudo","bash", "-c", f"mv {tmp_path} '{os.path.join(DIR,"preload_embeddings.pkl")}' && systemctl restart org.FaceRecognition"])
        print(f"Saved your face as {face_name} successfully. The daemon has been restarted and will be opperating in a few seconds.")
    except subprocess.CalledProcessError:
        print("Failed to save face!!! Maybe you don't have root permissions !")
    
def remove_face(*selection):
    username=os.environ["USER"]
    if len(selection)==0 or "all" in selection:
        input("Are you sure to delete all your saved faces ? They can't be restored. Type your root password to continue.")
        del ref_embeddings[username]
    else:
        for face_name in set(selection):
            try:
                del ref_embeddings[username][face_name]
            except IndexError:
                print(f"Unable to delete {face_name}, because it doesn't exists...")
    tmp_path="/tmp/facerec.tmp"
    print("Please enter your password in order to save your new face:")
    with open(tmp_path, "wb") as f:
        pickle.dump(ref_embeddings, f)
    try:
        subprocess.check_output(["sudo","bash", "-c", f"mv {tmp_path} '{os.path.join(DIR,"preload_embeddings.pkl")}' && systemctl restart org.FaceRecognition"])
        print(f"Deleted face{"s" if len(set(selection))>1 else ""} successfully. The daemon has been restarted and will be opperating in a few seconds.")
    except subprocess.CalledProcessError:
        print(f"Failed to delete face{"s" if len(set(selection))>1 else ""}!!! Maybe you don't have root permissions !")


if __name__=="__main__":
    if len(sys.argv)>1:
        if sys.argv[1]=="add":
            add_face(*sys.argv[2:])
        elif sys.argv[1]=="remove":
            remove_face(*sys.argv[2:])
        elif sys.argv[1]=="check":
            print(check(os.environ["USER"]))
    else:
        print(USAGE)
    
