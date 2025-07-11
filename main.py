import cv2
import numpy as np
import os
from ImProc_Detect_funcs import ImageProcessor
from datetime import date
from sklearn.cluster import DBSCAN
# Config
frame_step = 1
fire_imgs_folder = r"U:\Dissertation\Data\My First Project.v1i.yolov8\train\images"
# U:\Dissertation\Data\My First Project.v1i.yolov8\train\images - Aditya anotated training images of fire
# U:\Dissertation\Data\Binary\Binary\Unimodal\thermal_modality\train\Fire - THERMAL UNIMODAL
# U:\Dissertation\Data\Binary\Binary\Unimodal\rgb_modality\train\Fire     - RGB UNIMODAL
detection_folder = r"U:\Dissertation\Data\Detection & Segmentation output\Detected"
original_imgs_folder = r"U:\Dissertation\Data\Detection & Segmentation output\Original"
os.makedirs(detection_folder, exist_ok=True)
os.makedirs(original_imgs_folder, exist_ok=True)

# Detection parameters
alpha = 1.8
beta = 20
R_thresh = 120
B_thresh = 230
contour_area = 400

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
    # areas = []
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < contour_area:
    #         areas.append(area) # This was saving the rejected areas which you were then getting the centroid of  
    #         continue
    #     x, y, w, h = cv2.boundingRect(contour)
    #     frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < contour_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)  # green box

    if boxes:
        # 1. Find the centers of each box
        centers = np.array([[x + w / 2, y + h / 2] for x, y, w, h in boxes])

        # 2. Apply clustering (tweak eps as needed!)
        clustering = DBSCAN(eps=100, min_samples=1).fit(centers)

        # 3. Group boxes by cluster
        for cluster_id in set(clustering.labels_):
            cluster_boxes = [boxes[i] for i in range(len(boxes)) if clustering.labels_[i] == cluster_id]

            # 4. Merge boxes within each cluster
            xs = [x for x, y, w, h in cluster_boxes]
            ys = [y for x, y, w, h in cluster_boxes]
            x2s = [x + w for x, y, w, h in cluster_boxes]
            y2s = [y + h for x, y, w, h in cluster_boxes]

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(x2s), max(y2s)

            # 5. Draw one red box per cluster
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)

        
    # Save outputs
    impath = os.path.join(original_imgs_folder, f"Test{frame_count}.png")
    det_path = os.path.join(detection_folder, f"Test{frame_count}.png")
    cv2.imwrite(impath, original_frame)
    cv2.imwrite(det_path, frame)

    cv2.imshow("Detections", frame)
    cv2.imshow("Original", original_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
