#!/usr/bin/env python
import sys
import cv2
import numpy as np
from openvino import Core, convert_model
import os
import subprocess
import pickle
import time
import shutil
from skimage.feature import local_binary_pattern



DIR = os.path.dirname(__file__)
CAP_PATHS = ["/dev/video0","/dev/video2"] # fallback mechanism
CAPTURE_PATH = "/dev/video2"
RECOGNITION_TIMEOUT = 1.5
FACE_THRESHOLD = 0.7
IMPROVE_THRESHOLD = 0.8
FIRST_TRAIN_SIZE = 50
SPOOF_EXPECTANCE = 0.5
# Sum of K and C: Max threshold size (for 1st frame)
K = 0.2
# Min threshold
C = 0.5
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
compiled_det = core.compile_model(det_model, "AUTO")
# 2. Face Landmarks checking model
landmarks_model_path = os.path.join(DIR,"models/landmarks-regression-retail-0009.xml")
landmarks_model = core.read_model(landmarks_model_path)
compiled_landmarks = core.compile_model(landmarks_model, "AUTO")
# 3. Face Recognition Model
rec_model_path = os.path.join(DIR,"models/face-reidentification-retail-0095.xml")
rec_model = core.read_model(rec_model_path)
compiled_rec = core.compile_model(rec_model, "AUTO")
# 4. Anti-Spoof model
anti_spoof_model_path = os.path.join(DIR,"models/minifacenetv2.xml")
anti_spoof_model = core.read_model(anti_spoof_model_path)
compiled_anti_spoof = core.compile_model(anti_spoof_model, "AUTO")

def display_bgr_term(frame):
    def unicode_color_fg(b, g, r): return f"\x1b[38;2;{int(r)};{int(g)};{int(b)}m"
    def unicode_color_bg(b, g, r): return f"\x1b[48;2;{r};{g};{b}m"
    cols, rows = shutil.get_terminal_size()
    cols = min(cols,rows*2,64)
    rows = cols
    frame = cv2.resize(frame, (cols, rows))
    frame_unicode = "" #f"\x1b[{rows+1}A\r"
    for i in range(0, len(frame), 2):
        row1,row2=frame[i], frame[i+1]
        for pixtop,pixbottom in zip(row1,row2):
            frame_unicode+=unicode_color_bg(*pixbottom)
            frame_unicode+=unicode_color_fg(*pixtop)
            frame_unicode+="▄"
        frame_unicode+="\n"
    frame_unicode+="\x1b[0m" + f"\x1b[{rows//2}A\r"
    return frame_unicode

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

def stretch_contrast(img_bgr):
    # Split channels
    channels = cv2.split(img_bgr)
    stretched = []

    for ch in channels:
        min_val, max_val = np.min(ch), np.max(ch)
        # Avoid divide-by-zero
        if max_val > min_val:
            norm = ((ch - min_val) * 255.0 / (max_val - min_val)).astype(np.uint8)
        else:
            norm = ch
        stretched.append(norm)

    return cv2.merge(stretched)

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

def apply_gamma(img, gamma):
    """Apply gamma correction to an image (input range 0-255)"""
    # Build lookup table
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction
    return cv2.LUT(img, table)

def apply_clahe_bgr(img_bgr):
    # Convert BGR to YUV
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)

    # Apply CLAHE to the Y channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

    # Convert back to BGR
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_clahe

def adaptive_face_preprocessing(img_bgr, 
                               auto_gamma=True, 
                               edge_weight=0.1,
                               ir_mode=False):
    """
    Enhanced pipeline with:
    1. Dynamic gamma correction
    2. Adaptive histogram equalization
    3. Edge-aware normalization
    4. Cross-camera compatibility
    """
    img_bgr = stretch_contrast(img_bgr)
    # Convert to grayscale/Y-channel processing
    if ir_mode:
        # IR camera-specific processing
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    else:
        # RGB camera processing
        yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV)
        gray = yuv[:,:,0]

    # 1. Dynamic gamma correction
    if auto_gamma:
        gamma = calculate_adaptive_gamma(gray)
    else:
        gamma = 0.8 if ir_mode else 1.2  # Baseline values
    
    gamma_corrected = apply_gamma(img_bgr, gamma)

    # 2. Adaptive histogram equalization
    equalized = apply_clahe_bgr(gamma_corrected)

    # 3. Edge-aware normalization
    edges = optimized_canny(equalized, sigma=1.4, 
                          low_thresh=30, high_thresh=70)
    
    # 4. Edge-enhanced fusion
    final = edge_blend_bgr(equalized, edges, edge_weight)

    return final

def calculate_adaptive_gamma(gray_img):
    """Auto-determine gamma based on image brightness"""
    mean_brightness = np.mean(gray_img)
    return np.clip(0.5 + (mean_brightness/255), 0.3, 2.0)

def optimized_canny(img, sigma=1.4, low_thresh=30, high_thresh=70):
    """Improved Canny with noise-aware thresholds"""
    blurred = cv2.GaussianBlur(img, (5,5), sigma)
    grad_x = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=3)
    
    # Dynamic thresholding based on gradient magnitudes
    mag = np.sqrt(grad_x**2 + grad_y**2)
    non_zero = mag[mag > 0]
    if len(non_zero) > 0:
        high = np.percentile(non_zero, 95)
        low = high * 0.5
        return cv2.Canny(blurred, low, high)
    return np.zeros_like(img)

def edge_blend_bgr(img_bgr, edges, edge_weight=0.3):
    """
    Blends grayscale edge map into each BGR channel.
    Assumes edges is normalized to [0, 1] or [0, 255]
    """
    # Normalize edges to 0–1
    if edges.max() > 1:
        edges = edges.astype(np.float32) / 255.0

    # Convert BGR to float
    img_bgr = img_bgr.astype(np.float32) / 255.0

    # Expand edges to 3 channels
    edges_3ch = np.stack([edges]*3, axis=-1)

    # Blend
    blended = (1 - edge_weight) * img_bgr + edge_weight * edges_3ch
    blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

    return blended

def image_quality_score(img_bgr,
                        w_sharp=0.4,
                        w_contrast=0.4,
                        w_noise=0.2,
                        sharp_thresh=(100, 1000),
                        contrast_thresh=(10, 70),
                        noise_thresh=(0.01, 0.1)):
    """
    Compute a quality score in [0,1] for a BGR image.

    Parameters
    ----------
    img_bgr : np.ndarray
        Input image in BGR (uint8).
    w_sharp, w_contrast, w_noise : float
        Weights for sharpness, contrast, and noise in the final score.
        Must sum to 1.0.
    sharp_thresh : (low, high)
        Expected min/max of Laplacian variance for good focus.
    contrast_thresh : (low, high)
        Expected min/max of RMS contrast for good lighting.
    noise_thresh : (low, high)
        Expected min/max of estimated noise std for acceptable noise levels.

    Returns
    -------
    score : float
        Quality score between 0 (worst) and 1 (best).
    """
    # 1) Sharpness: variance of Laplacian
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    var_lap = lap.var()
    # normalize to [0,1]
    s = np.clip((var_lap - sharp_thresh[0]) / (sharp_thresh[1] - sharp_thresh[0]), 0, 1)
    
    # 2) Contrast: RMS contrast = std(gray)
    rms_contrast = gray.std()
    c = np.clip((rms_contrast - contrast_thresh[0]) / (contrast_thresh[1] - contrast_thresh[0]), 0, 1)
    
    # 3) Noise: estimate via patch-based standard deviation of high-freq residual
    #    subtract a 3×3 mean filter to isolate noise
    blur = cv2.blur(gray, (3,3))
    residual = gray.astype(np.float32) - blur.astype(np.float32)
    noise_std = residual.std() / 255.0  # normalize to [0,1]
    # lower noise_std is better, so invert and normalize
    n = np.clip((noise_thresh[1] - noise_std) / (noise_thresh[1] - noise_thresh[0]), 0, 1)
    
    # Combine
    score = w_sharp*s + w_contrast*c + w_noise*n
    return float(score)

def check_light(image): # don't try if laptop camera is obstructed
    to_check = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return False if to_check.mean()<20 else True

# --- Utility: Cosine Similarity ---
def cosine_similarity(a, b):
    """Check the cosine similarity of two vertex arrays"""
    a = a.flatten()  # Convert to 1D
    b = b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def get_scaled_ref_landmarks(output_size=(112, 112), zoom=1.0, widen=1.2):
    """
    Create reference landmarks scaled to fill the canvas more horizontally.
    
    :param output_size: Size of the aligned face image.
    :param zoom: Overall scaling of the face.
    :param widen: Horizontal stretch factor (1.0 = no change, >1 = wider face).
    :return: Transformed reference landmarks.
    """
    base = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.1396, 92.3655], # mid mouth
    ], dtype=np.float32)

    center = np.mean(base, axis=0)

    # Apply zoom (both axes)
    scaled = (base - center) * zoom

    # Widen horizontally
    scaled[:, 0] *= widen

    # Translate to output center
    scaled += np.array(output_size) / 2.0

    return scaled

def align_face_with_landmarks(face_bgr, orig_frame, bbox, output_size=(128, 128),
                              ref_landmarks=None, anti_spoof_size_boost = 2.7):
    """
    Aligns a face image using landmarks-regression-retail-0009 and OpenCV.

    Parameters:
        compiled_landmarks: OpenVINO CompiledModel for landmarks-regression-retail-0009.
        face_bgr (np.ndarray): Cropped face in BGR format.
        output_size (tuple): Desired aligned output size (width, height).
        ref_landmarks (np.ndarray, optional): Custom 5×2 reference points in output coords.
            If None, default ArcFace reference points are used.

    Returns:
        np.ndarray: Aligned face image of shape (output_size[1], output_size[0], 3).
    """
    output = adaptive_face_preprocessing(face_bgr)
    #output=face_bgr
    # 1. Get model I/O information
    input_info = compiled_landmarks.inputs[0]
    output_info = compiled_landmarks.outputs[0]
    _, _, H, W = input_info.shape
    input_name = input_info.any_name
    output_name = output_info.any_name

    # 2. Preprocess: resize to model input (48×48), maintain BGR, pack as NCHW
    resized = cv2.resize(face_bgr, (W, H))
    tensor = resized.transpose(2, 0, 1).astype(np.float32)[np.newaxis, :]

    # 3. Inference
    results = compiled_landmarks({input_name: tensor})
    flat = results[output_name].flatten()               # shape: (10,)
    landmarks = flat.reshape((5, 2)).astype(np.float32) # normalized coords

    # 4. Scale to pixel coordinates in original crop
    h, w = face_bgr.shape[:2]
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h
    landmarks += (bbox[0],bbox[1])
    mouth_coords = landmarks[3:]
    mouth_center = (mouth_coords[0,0]+mouth_coords[1,0])/2,(mouth_coords[0,1]+mouth_coords[1,1])/2
    landmarks = np.array([*landmarks[:2],mouth_center])[:2]
    # 5. Reference points (ArcFace), scaled to output_size
    ref_landmarks = get_scaled_ref_landmarks((128,128),zoom=1)[:2] # + (bbox[0],bbox[1])
    ref_landmarks_anti_spoof = get_scaled_ref_landmarks((128,128),zoom=1/anti_spoof_size_boost)[:2]

    # 6. Estimate affine transform
    tform, _ = cv2.estimateAffinePartial2D(landmarks, ref_landmarks, method=cv2.LMEDS)
    tform_spoof, _ = cv2.estimateAffinePartial2D(landmarks, ref_landmarks_anti_spoof, method=cv2.LMEDS)
    # 7. Warp the original BGR crop to the aligned output
    aligned = cv2.warpAffine(orig_frame, tform, output_size, flags=cv2.INTER_LINEAR)
    anti_spoof_aligned = cv2.warpAffine(orig_frame, tform_spoof, output_size, flags=cv2.INTER_LINEAR)
    
    return aligned, anti_spoof_aligned


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
def parse_detections(detections, frame_shape, conf_threshold=0.5, minsize = 80):
    h, w = frame_shape[:2]
    boxes = []
    # Assuming the detection model output is in the format:
    # [image_id, label, conf, xmin, ymin, xmax, ymax] for each detection.
    for detection in detections[0][0]:
        confidence = float(detection[2])
        if confidence < conf_threshold:
            continue
        xmin = int(detection[3] * w)
        ymin = int(detection[4] * h)
        xmax = int(detection[5] * w)
        ymax = int(detection[6] * h)
        if xmax - xmin < minsize or ymax - ymin < minsize:
            continue
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
    current_cap=0
    failed_find_attempts=[0 for c in CAP_PATHS]
    if not username in ref_embeddings:
        return "fail"
    if len(ref_embeddings[username])==0:
        return "Error: training base is too small"
    cap = cv2.VideoCapture(CAP_PATHS[current_cap], cv2.CAP_V4L2)
    if not cap.isOpened():
        return "Error: Failed to open Webcam"
    attempts_count=0
    ret, frame = cap.read()
    start_time = time.monotonic()
    list_faces = []
    spoof_attempts = 0
    reason  = "not recognized"
    def score(rec_embedding):
        return max(cosine_similarity(ref_emb, rec_embedding) for ref_emb in user_face for key, user_face in ref_embeddings[username].items())
    while attempts_count<n_try and any(i<11 for i in failed_find_attempts) and time.monotonic()<start_time+RECOGNITION_TIMEOUT:
        did_try=0
        if spoof_attempts>2:
            reason = "spoof detected"
            break
        if failed_find_attempts[current_cap] > 10:
            current_cap = (current_cap + 1) % len(CAP_PATHS)
            cap.release()
            cap = cv2.VideoCapture(CAP_PATHS[current_cap])
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frame = ensure_bgr(frame)
        

        if not image_quality_score(frame)>0.3:
            failed_find_attempts[current_cap] += 1
            continue
        else:
            failed_find_attempts[current_cap] //= 2
        
        # 1. Detect faces using the detection model
        # Preprocess frame for detection (resize to model's expected input, e.g., 300x300)
        det_input_size = (300, 300)  # Adjust if needed
        frame_resized = cv2.resize(frame, det_input_size)
        # Assume detection model requires CHW; adjust conversion if needed
        input_blob_det = frame_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        
        # Run detection
        det_result = compiled_det([input_blob_det])[compiled_det.output(0)]
        # Parse detection output (update parsing based on your model’s output format)
        boxes = parse_detections(det_result, frame.shape, conf_threshold=FACE_THRESHOLD)
        for_loop_list_faces = []
        for (xmin, ymin, xmax, ymax, conf) in boxes:
            # Crop the detected face from the original frame
            face_crop = frame[ymin:ymax, xmin:xmax]
            if not image_quality_score(face_crop) > 0.3:
                continue
            did_try=1
            # 3. Run recognition on the face crop
            #rec_face = apply_clahe(face_crop)
            rec_face = stretch_contrast(face_crop)
            rec_face, anti_spoof_face = align_face_with_landmarks(rec_face, frame, (xmin,ymin,xmax,ymax))
            #rec_face = gray_world_correction(rec_face)
            anti_spoof_face = cv2.resize(anti_spoof_face,[80,80]).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            anti_spoof_result = compiled_anti_spoof([anti_spoof_face])[compiled_anti_spoof.output(0)]
            label = np.argmax(anti_spoof_result)
            value = anti_spoof_result[0][label]/2
            
            if label != 1:
                spoof_attempts+=1
                break

            rec_input = adaptive_face_preprocessing(rec_face).transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            rec_embedding = compiled_rec([rec_input])[compiled_rec.output(0)]
            # Compare with every reference embedding
            for key, user_face in ref_embeddings[username].items():
                similarities = [cosine_similarity(ref_emb, rec_embedding) for ref_emb in user_face]
                sim = max(similarities)
                if sim > C:
                    for_loop_list_faces.append((rec_embedding, key, anti_spoof_face))
                    break
        if len(for_loop_list_faces):
            list_faces.append(max(for_loop_list_faces,key=lambda k:score(k[0])))
            threshold = K / len(list_faces) + C
                
            if all(score(rec[0])>threshold for rec in list_faces):
                cap.release()
                for rec in list_faces:
                    if score(rec[0])<IMPROVE_THRESHOLD and len(ref_embeddings[username][rec[1]])<500:
                        ref_embeddings[username][rec[1]].append(rec[0])
                if any([score(rec[0]) < IMPROVE_THRESHOLD for rec in list_faces]): # use as training image
                    try:
                        with open(os.path.join(DIR,"preload_embeddings.pkl"), "wb") as f:
                            pickle.dump(ref_embeddings, f)
                    except PermissionError:
                        pass
                return "pass"
        if did_try == 0: # if there were no boundings of a sufficient quality, then this try wasn't good for this cap
            failed_find_attempts[current_cap] += 1
        attempts_count+=did_try
        time.sleep(0.1)
    cap.release()
    return f"fail - {reason}"
    
def add_face(cap_path=...,face_name=...,complete=False):
    """
    python facerec.py add
    Allow to save needed data for good performance on first unlock attempts.
    The recognition will then be trained more with successfull recognitions that are not too much similar to the original one.
    
    !WARN! That can be unsafe, as if your camera takes too bad photos, this can help someone else to get recognized more easely.
    
    But in most cases, it just improve performance (you need less frames to be recognized).
    """
    total_progress=0
    if cap_path==... or not os.path.exists(cap_path):
        cap_path=CAP_PATHS[0]
    username = os.environ["USER"]
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
    cap = cv2.VideoCapture(cap_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        return "Error: Failed to open Webcam"
    while True:
        appended=False
        ret, frame = cap.read()
        if not ret:
            break
        frame = ensure_bgr(frame)
        h,w,c = frame.shape
        
        # 1. Detect faces using the detection model
        # Preprocess frame for detection (resize to model's expected input, e.g., 300x300)
        det_input_size = (300, 300)  # Adjust if needed
        frame_resized = cv2.resize(frame, det_input_size)
        # Assume detection model requires CHW; adjust conversion if needed
        input_blob_det = frame_resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        # Run detection
        det_result = compiled_det([input_blob_det])[compiled_det.output(0)]
        # Parse detection output (update parsing based on your model’s output format)
        boxes = parse_detections(det_result, frame.shape, conf_threshold=FACE_THRESHOLD)
        appened=False
        for (xmin, ymin, xmax, ymax, conf) in boxes:
            xmin, ymin = max(xmin - 10,0), max(ymin - 10,0) # Leave some margin to get a usable result after re-alignment 
            xmax, ymax = min(xmax + 10,w), min(ymax + 10,h)
            # Draw box
            #cv2.rectangle(frame, (xmin-2, ymin-2), (xmax+2, ymax+2), (0, 255, 0), 2)
            # Crop the detected face from the original frame
            face_crop = frame[ymin:ymax, xmin:xmax]
            crop_dims=xmax-xmin,ymax-ymin
            if not image_quality_score(face_crop)>0.3:
                continue # skip unusable faces crop
            # Vertically align the face crop using landmarks detection
            rec_face = stretch_contrast(face_crop)
            rec_face , anti_spoof_face = align_face_with_landmarks(rec_face, frame, (xmin,ymin,xmax,ymax))
            #rec_face=gray_world_correction(rec_face)
            frame[ymin:ymax, xmin:xmax] = cv2.resize(rec_face,crop_dims)
            # 3. Run recognition on the face crop
            rec_input = rec_face.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
            rec_embedding = compiled_rec([rec_input])[compiled_rec.output(0)]
            # Compare with every reference embedding
            similarities = [cosine_similarity(ref_emb, rec_embedding) for ref_emb in new_face]
            face_preview = cv2.resize(rec_face, (128, 128))
            # You can adjust your threshold accordingly
            if username == "root":
                print(display_bgr_term(rec_face),end="")
            if all([0.2 < sim < IMPROVE_THRESHOLD for sim in similarities]) or len(new_face)==0: # use as training image
                new_face.append(rec_embedding)
                appened=True
        if appened:
            quality_score = image_quality_score(frame)
            if not complete:
                total_progress+=((quality_score)*4.5) 
            else:
                total_progress+=((quality_score)*2.5)

        text = f"Face training : {total_progress:.2f} %"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        bar_x, bar_y = 10, 50
        bar_w = 550
        if not username == "root":
            print(total_progress)
        progress = int(total_progress/100 * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + 20), (0, 255, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (255, 255, 255), 2)
        if not username == "root":
            cv2.imshow("Webcam - Press 's' to save face", frame)
        else:
            print(" " + str(int(total_progress))+"%",end="\r")
        appended=False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if total_progress>=100:
            print("Succeeded to record needed data...\nYour face has been recorded.")
            print(len(new_face),"images registered successfully.")

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
            add_face(*sys.argv[2:],complete=False)
        elif sys.argv[1]=="complete_add":
            add_face(*sys.argv[2:],complete=True)
        elif sys.argv[1]=="remove":
            remove_face(*sys.argv[2:])
        elif sys.argv[1]=="check":
            print(check(os.environ["USER"]))
    else:
        print(USAGE)
    