import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'   # Fix for OMP Error #15 on Windows

import sys
import argparse
import glob
import time
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file', required=True)
parser.add_argument('--source', help='Image source: image / folder / video / usb0 / picamera0', required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold', default=0.5)
parser.add_argument('--resolution', help='Resolution WxH (example: 640x480)', default=None)
parser.add_argument('--record', help='Record to demo1.avi (needs --resolution)', action='store_true')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

print(f'[INFO] thresh = {min_thresh}')
print(f'[INFO] model  = {model_path}')
print(f'[INFO] source = {img_source}')
print(f'[INFO] cwd    = {os.getcwd()}')

if not os.path.exists(model_path):
    print('[ERROR] Model path is invalid or model was not found.')
    sys.exit(0)

# ---------------- Load model ----------------
model = YOLO(model_path, task='detect')
labels = model.names
print(f'[INFO] Model loaded. Classes: {labels}')

# ---------------- Determine source type ----------------
img_ext_list = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP']
vid_ext_list = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'[ERROR] File extension {ext} not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'[ERROR] Input {img_source} is invalid.')
    sys.exit(0)

print(f'[INFO] source_type = {source_type}')

# ---------------- Resolution ----------------
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])
    print(f'[INFO] target resolution = {resW}x{resH}')

# ---------------- Recording setup ----------------
if record:
    if source_type not in ['video', 'usb']:
        print('[ERROR] Recording only works for video and camera sources.')
        sys.exit(0)
    if not user_res:
        print('[ERROR] Please specify resolution to record video at.')
        sys.exit(0)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'),
                               record_fps, (resW, resH))

# ---------------- Load source ----------------
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    for file in glob.glob(img_source + '/*'):
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
    print(f'[INFO] Found {len(imgs_list)} images in folder.')
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if not cap.isOpened():
        print('[ERROR] Cannot open camera/video source.')
        sys.exit(0)
    print('[INFO] Camera/video opened OK.')
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
        print(f'[INFO] Actual cam res: {int(cap.get(3))}x{int(cap.get(4))}')
elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'RGB888', "size": (resW, resH)}))
    cap.start()

# ---------------- Bounding box colors ----------------
bbox_colors = [(164, 120, 87), (68, 148, 228), (93, 97, 209), (178, 182, 133),
               (88, 159, 106), (96, 202, 231), (159, 124, 168), (169, 162, 241),
               (98, 118, 150), (172, 176, 184)]

# ---------------- Save folder ----------------
save_dir = os.path.join(os.getcwd(), 'face_save')
os.makedirs(save_dir, exist_ok=True)
print(f'[INFO] Save dir = {save_dir}')
print(f'[INFO] Save dir exists? {os.path.isdir(save_dir)}')
print(f'[INFO] Save dir writable? {os.access(save_dir, os.W_OK)}')

# Write a test image to confirm saving works at OS level
test_path = os.path.join(save_dir, '_write_test.jpg')
test_ok = cv2.imwrite(test_path, np.zeros((50, 50, 3), dtype=np.uint8))
print(f'[INFO] Test write to face_save/: {test_ok}  ->  {test_path}')
if test_ok:
    try:
        os.remove(test_path)
    except Exception:
        pass

# Cooldown: save at most once per class every N seconds
last_save_time = {}
SAVE_COOLDOWN_SEC = 2.0
save_count = 0
frame_count = 0

# FPS tracking
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# ---------------- Main loop ----------------
while True:
    t_start = time.perf_counter()
    frame_count += 1

    # ---- Read frame ----
    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print('[INFO] All images have been processed. Exiting.')
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type == 'video':
        ret, frame = cap.read()
        if not ret:
            print('[INFO] Reached end of the video file.')
            break
    elif source_type == 'usb':
        ret, frame = cap.read()
        if frame is None or not ret:
            print('[ERROR] Cannot read from camera.')
            break
    elif source_type == 'picamera':
        frame = cap.capture_array()
        if frame is None:
            print('[ERROR] Cannot read from Picamera.')
            break

    if frame is None:
        print('[WARN] frame is None, skipping.')
        continue

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    # ---- Inference ----
    results = model(frame, verbose=False)
    detections = results[0].boxes
    object_count = 0

    # ---- Loop detections ----
    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        h, w = frame.shape[:2]
        xmin = max(0, min(xmin, w - 1))
        xmax = max(0, min(xmax, w))
        ymin = max(0, min(ymin, h - 1))
        ymax = max(0, min(ymax, h))

        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = float(detections[i].conf.item())

        if conf > min_thresh:
            # ---- Draw bounding box ----
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(frame,
                          (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10),
                          color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # ---- Save crop ----
            now = time.time()
            cooldown_ok = (now - last_save_time.get(classname, 0)) > SAVE_COOLDOWN_SEC

            if cooldown_ok:
                # .copy() is CRITICAL — YOLO frames can be non-contiguous,
                # which makes cv2.imwrite silently return False
                crop = frame[ymin:ymax, xmin:xmax].copy()

                if crop.size > 0:
                    # Ensure contiguous uint8 (belt-and-braces)
                    crop_save = np.ascontiguousarray(crop).astype(np.uint8)

                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    crop_filename = os.path.join(
                        save_dir,
                        f'{classname}_{timestamp}_{save_count:05d}.jpg'
                    )

                    try:
                        ok = cv2.imwrite(crop_filename, crop_save)
                    except Exception as e:
                        print(f'[ERROR] imwrite threw exception: {e}')
                        ok = False

                    # Fallback via imencode + open() if imwrite failed
                    if not ok:
                        print(f'[WARN] imwrite failed for {crop_filename}, trying fallback...')
                        try:
                            success, buf = cv2.imencode('.jpg', crop_save)
                            if success:
                                with open(crop_filename, 'wb') as f:
                                    f.write(buf.tobytes())
                                ok = True
                                print(f'[FALLBACK] Wrote file via imencode: {crop_filename}')
                        except Exception as e:
                            print(f'[ERROR] Fallback failed: {e}')

                    if ok:
                        save_count += 1
                        print(f'[SAVE] OK -> {crop_filename}  crop shape={crop_save.shape}')
                    else:
                        print(f'[SAVE] FAILED -> {crop_filename}')
                else:
                    print('[WARN] crop was empty, skipped.')

                # Update cooldown regardless of success to avoid spamming
                last_save_time[classname] = now

            object_count += 1

    # ---- Status text ----
    if source_type in ['video', 'usb', 'picamera']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:0.2f}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    cv2.putText(frame, f'Objects: {object_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
    cv2.putText(frame, f'Saved: {save_count}', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)

    cv2.imshow('YOLO detection results', frame)
    if record:
        recorder.write(frame)

    # ---- Keys ----
    if source_type in ['image', 'folder']:
        key = cv2.waitKey()
    else:
        key = cv2.waitKey(5)

    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('capture.png', frame)

    # ---- FPS ----
    t_stop = time.perf_counter()
    frame_rate_calc = float(1 / (t_stop - t_start))
    if len(frame_rate_buffer) >= fps_avg_len:
        frame_rate_buffer.pop(0)
    frame_rate_buffer.append(frame_rate_calc)
    avg_frame_rate = np.mean(frame_rate_buffer)

# ---------------- Cleanup ----------------
print(f'[INFO] Done. Frames processed: {frame_count}, crops saved: {save_count}')
print(f'[INFO] Average pipeline FPS: {avg_frame_rate:.2f}')

if source_type in ['video', 'usb']:
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record:
    recorder.release()
cv2.destroyAllWindows()