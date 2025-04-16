import sys
import cv2
import numpy as np
from openvino import Core
import os
import pickle
import time
DIR = os.path.dirname(__file__)
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
elif os.path.exists(os.path.join(DIR,"images")):
    # Fallback: compute and save [deprecated because anything should be in the pkl preload]
    ref_images = [cv2.imread(os.path.join(DIR,f"/images/{file}")) for file in os.listdir(os.path.join(DIR,"images"))]
    ref_faces = [cv2.resize(ref_image, (128, 128)) for ref_image in ref_images]
    ref_faces = [ref_face.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) for ref_face in ref_faces]
    ref_embeddings = [compiled_rec([ref_face])[compiled_rec.output(0)] for ref_face in ref_faces]

    with open(os.path.join(DIR,"preload_embeddings.pkl"), "wb") as f: # save the preload
        pickle.dump(ref_embeddings, f)
else:
    ref_embeddings = []
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

def check(n_try=5):
    """
    Deserves to the face recognition in itself.
    Called by the daemon when it receives an auth request
    take as argument the numbers of frames to check, spaced by ~0.1 second
    """
    if len(ref_embeddings)<2:
        return "Error: training base is too small"
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Failed to open Webcam"
    for i in range(n_try):
        ret, frame = cap.read()
        if not ret:
            break
        frame=cv2.filter2D(frame,-1,np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        frame = cv2.medianBlur(frame,3)
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
            if face_crop.size == 0:
                continue
            # 3. Run recognition on the face crop
            rec_face = cv2.resize(face_crop, (128, 128))
            rec_input = rec_face.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            rec_embedding = compiled_rec([rec_input])[compiled_rec.output(0)]
            # Compare with every reference embedding
            similarities = [cosine_similarity(ref_emb, rec_embedding) for ref_emb in ref_embeddings]
            # You can adjust your threshold accordingly
            is_match = any([sim > 0.7 for sim in similarities])
            if is_match:
                cap.release()
                if all([sim < 0.8 for sim in similarities]) and len(ref_embeddings)<500: # use as training image
                    ref_embeddings.append(rec_embedding)
                    with open(os.path.join(DIR,"preload_embeddings.pkl"), "wb") as f:
                        pickle.dump(ref_embeddings, f)
                return "pass"
        time.sleep(0.1)
    cap.release()
    return "fail"
    
    
def add_face():
    """
    python facerec.py add
    Allow to save needed data for good performance on first unlock attempts.
    The recognition will then be trained more with successfull recognitions that are not too much similar to the original one.
    
    !WARN! That can be unsafe, as if your camera takes too bad photos, this can help someone else to get recognized more easely.
    
    But in most cases, it just improve performance.
    """
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Failed to open Webcam"
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame=cv2.filter2D(frame,-1,np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        frame = cv2.medianBlur(frame,3)
        
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
            if face_crop.size == 0:
                continue
            # 3. Run recognition on the face crop
            rec_face = cv2.resize(face_crop, (128, 128))
            rec_input = rec_face.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            rec_embedding = compiled_rec([rec_input])[compiled_rec.output(0)]
            # Compare with every reference embedding
            similarities = [cosine_similarity(ref_emb, rec_embedding) for ref_emb in ref_embeddings]
            face_preview = cv2.resize(rec_face, (128, 128))
            cv2.imshow("Face Crop", rec_face)
            # You can adjust your threshold accordingly
            if (all([0.4 < sim < 0.8 for sim in similarities]) and len(ref_embeddings)<50) or len(ref_embeddings)==0: # use as training image
                ref_embeddings.append(rec_embedding)
                print(len(ref_embeddings),"/",50,"training frames saved...")
        cv2.imshow("Webcam - Press 's' to save face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(ref_embeddings)==50:
            print("Succeeded to record needed data...\nYour face has been setup.")
            break
                
    cap.release()
    # save faces as vertex data (safer than images)
    with open(os.path.join(DIR,"preload_embeddings.pkl"), "wb") as f:
        pickle.dump(ref_embeddings, f)
        
if __name__=="__main__":
    if sys.argv[1]=="add":
        add_face()
