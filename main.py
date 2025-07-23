import cv2
import numpy as np
import os
from ImProc_Detect_funcs import ImageProcessor
from datetime import date
from sklearn.cluster import DBSCAN
# Config
frame_step = 1
fire_imgs_folder = r"U:\Dissertation\Data\yolov8_non_augmentation\train\images"
# U:\Dissertation\Data\yolov8_augmentation\train\images - Aditya fire images AUGMENTATED
# U:\Dissertation\Data\yolov8_non_augmentation\train\images - Aditya fire images NO augmentation
# U:\Dissertation\Data\Binary\Binary\Unimodal\thermal_modality\train\Fire - THERMAL UNIMODAL
# U:\Dissertation\Data\Binary\Binary\Unimodal\rgb_modality\train\Fire     - RGB UNIMODAL
detection_folder = r"U:\Dissertation\Data\Detection & Segmentation output\Detected"
original_imgs_folder = r"U:\Dissertation\Data\Detection & Segmentation output\Original"
os.makedirs(detection_folder, exist_ok=True)
os.makedirs(original_imgs_folder, exist_ok=True)

# Detection parameters
alpha = 0.5
beta =40
R_thresh = 50
B_thresh = 230
contour_area = 250

# Get all image files
image_files = sorted([
    f for f in os.listdir(fire_imgs_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

# Process each image
for frame_count, image_file in enumerate(image_files):
    if frame_count % frame_step != 0:
        continue

    img_path = os.path.join(fire_imgs_folder, image_file)
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Could not read {img_path}")
        continue

    frame = cv2.resize(frame, (640, 512))
    original_frame = frame.copy()

    ImProc = ImageProcessor(frame)
    ImProc.preprocessor()
    ImProc.vbi_idx()
    ImProc.fi_idx()
    ImProc.ffi_idx(alpha)
    ImProc.calc_tf(alpha)
    ImProc.ffi_binarize()
    ImProc.erosion()
    ImProc.dilation()
    ffi_blurred = ImProc.blur().astype(np.uint8)
    D_f = cv2.bitwise_and(frame, frame, mask=ffi_blurred)

    rule1_result = ImProc.rule_1(beta).astype(np.uint8)
    rule2_result = ImProc.rule_2(R_thresh, B_thresh).astype(np.uint8)
    rule3_result = ImProc.rule_3().astype(np.uint8)
    D_s = cv2.bitwise_and(rule1_result, rule2_result)
    D_s = cv2.bitwise_and(rule3_result, D_s)

    D_fs_bin = cv2.bitwise_or(ffi_blurred, D_s)
    D_fs = cv2.bitwise_and(frame, frame, mask=D_fs_bin)
    E_result_raw = ImProc.wavelet_transform(D_fs, frame)
    E_result = cv2.resize(E_result_raw, (frame.shape[1], frame.shape[0]))  # (same width and heightwidth, height)


    D_result = cv2.bitwise_and(D_fs_bin, E_result).astype(np.uint8)
    final_result = cv2.bitwise_and(frame, frame, mask=D_result)

    contours, _ = cv2.findContours(D_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Segmentation Attempt
    # Create a color overlay from the binary mask
    # --- Fill fire regions (enclose all touching fire pixels) ---
    # Make sure mask is binary and uint8
    mask = D_result.copy()

    # Find contours of connected fire regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask and fill each contour
    filled_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > contour_area:  # Skip small specks
            cv2.drawContours(filled_mask, [cnt], -1, 255, thickness=cv2.FILLED)

    # Optional: smooth the boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filled_mask = cv2.dilate(filled_mask, kernel, iterations=1)

    # Create the red overlay
    colored_mask = np.zeros_like(frame)
    colored_mask[filled_mask > 0] = [0, 0, 255]
    cv2.imshow("coloured mask",colored_mask)
    segmentation_overlay = cv2.addWeighted(original_frame, 0.7, colored_mask, 0.3, 0)
        
    # Save outputs
    impath = os.path.join(original_imgs_folder, f"Test{frame_count}.png")
    det_path = os.path.join(detection_folder, f"Test{frame_count}.png")
    cv2.imwrite(impath, original_frame)
    cv2.imwrite(det_path, segmentation_overlay)

    cv2.imshow("Detections", segmentation_overlay)
    cv2.imshow("Original", original_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
