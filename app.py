import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import time
import threading
import queue
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS

# ─────────────────────────── Paths ───────────────────────────
BASE_DIR      = Path(__file__).parent
FACE_DIR      = BASE_DIR / 'registered_face'
FACE_SAVE_DIR = BASE_DIR / 'face_save'
DB_PATH       = BASE_DIR / 'people.json'
LOG_PATH      = BASE_DIR / 'recognition_log.json'
STATIC_DIR    = BASE_DIR / 'static'

FACE_DIR.mkdir(exist_ok=True)
FACE_SAVE_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ─────────────────────────── App ───────────────────────────
app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path='/static')
CORS(app)

# ─────────────────────────── DB helpers ───────────────────────────
def load_db():
    if DB_PATH.exists():
        with open(DB_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_db(data):
    with open(DB_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_log():
    if LOG_PATH.exists():
        with open(LOG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def append_log(entry):
    logs = load_log()
    logs.insert(0, entry)
    logs = logs[:500]
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

# ─────────────────────────── Encoding cache ───────────────────────────
known_encodings = []
known_ids       = []

def rebuild_encodings():
    global known_encodings, known_ids
    people = load_db()
    enc_list, id_list = [], []
    for p in people:
        for enc in p.get('encodings', []):
            enc_list.append(np.array(enc))
            id_list.append(p['id'])
    known_encodings = enc_list
    known_ids       = id_list
    print(f'[ENC] Rebuilt: {len(enc_list)} encodings for {len(set(id_list))} people')

rebuild_encodings()

# ─────────────────────────── Camera config ───────────────────────────
camera_config = {
    'source_type':  'webcam',
    'webcam_index': 0,
    'rtsp_url':     'rtsp://admin:admin@192.168.1.100:554/stream1',
}

# ─────────────────────────── Camera state ───────────────────────────
camera_lock    = threading.Lock()
camera_cap     = None
camera_running = False
recognition_queue = queue.Queue(maxsize=20)

TOLERANCE    = 0.50
COOLDOWN_SEC = 5.0
last_seen    = {}

# ══════════════════════════════════════════════════════════════════════
#  ZERO-LAG FRAME BUFFER
# ══════════════════════════════════════════════════════════════════════
_latest_frame   = None
_latest_seq     = 0            # increments every time a NEW frame arrives
_frame_lock     = threading.Lock()
_reader_active  = False
_reader_thread  = None

def _reader_loop(cap):
    global _latest_frame, _latest_seq, _reader_active
    while _reader_active:
        ret, frame = cap.read()
        if ret and frame is not None:
            with _frame_lock:
                _latest_frame = frame
                _latest_seq  += 1
        else:
            time.sleep(0.005)

def _start_reader(cap):
    global _reader_thread, _reader_active, _latest_frame, _latest_seq
    _reader_active = False
    if _reader_thread and _reader_thread.is_alive():
        _reader_thread.join(timeout=2)
    _latest_frame = None
    _latest_seq   = 0
    _reader_active = True
    _reader_thread = threading.Thread(target=_reader_loop, args=(cap,), daemon=True)
    _reader_thread.start()

def _stop_reader():
    global _reader_active
    _reader_active = False

def _grab_latest(timeout=3.0):
    """Return (seq, frame_copy) — only copy when a frame exists."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _frame_lock:
            f   = _latest_frame
            seq = _latest_seq
        if f is not None:
            return seq, f.copy()
        time.sleep(0.01)
    return None, None

# ══════════════════════════════════════════════════════════════════════
#  ASYNC RECOGNITION WORKER
#  Face detection + encoding runs in its own thread so the stream
#  thread is NEVER blocked by heavy CPU work.
# ══════════════════════════════════════════════════════════════════════
_recog_input   = queue.Queue(maxsize=1)   # frame to analyse
_recog_output  = queue.Queue(maxsize=1)   # list of (box, name, color) results
_recog_active  = False
_recog_thread  = None

def _recognition_worker():
    global _recog_active
    while _recog_active:
        try:
            frame = _recog_input.get(timeout=0.5)
        except queue.Empty:
            continue

        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs  = face_recognition.face_locations(rgb, model='hog')
        encs  = face_recognition.face_encodings(rgb, locs)

        people_db  = load_db()
        person_map = {p['id']: p for p in people_db}

        results = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            box   = (top*2, right*2, bottom*2, left*2)
            name  = 'Unknown'
            color = (0, 0, 220)

            if known_encodings:
                distances = face_recognition.face_distance(known_encodings, enc)
                best_idx  = int(np.argmin(distances))
                if distances[best_idx] <= TOLERANCE:
                    pid    = known_ids[best_idx]
                    person = person_map.get(pid, {})
                    name   = person.get('name', 'Unknown')
                    color  = (0, 210, 80)

                    now = time.time()
                    if now - last_seen.get(pid, 0) >= COOLDOWN_SEC:
                        last_seen[pid] = now
                        entry = {
                            'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'person_id':  pid,
                            'name':       person.get('name', ''),
                            'emp_id':     person.get('emp_id', ''),
                            'department': person.get('department', ''),
                            'confidence': round((1 - float(distances[best_idx])) * 100, 1),
                            'photo':      person.get('photo', ''),
                        }
                        append_log(entry)
                        try:
                            recognition_queue.put_nowait(entry)
                        except queue.Full:
                            pass

            results.append((box, name, color))

        # Replace any stale result
        try:
            _recog_output.get_nowait()
        except queue.Empty:
            pass
        try:
            _recog_output.put_nowait(results)
        except queue.Full:
            pass

def _start_recog_worker():
    global _recog_thread, _recog_active
    _recog_active = False
    if _recog_thread and _recog_thread.is_alive():
        _recog_thread.join(timeout=2)
    _recog_active = True
    _recog_thread = threading.Thread(target=_recognition_worker, daemon=True)
    _recog_thread.start()

def _stop_recog_worker():
    global _recog_active
    _recog_active = False

# ══════════════════════════════════════════════════════════════════════
#  CAMERA OPEN / CLOSE
# ══════════════════════════════════════════════════════════════════════
def get_camera():
    global camera_cap
    if camera_cap is None or not camera_cap.isOpened():
        if camera_config['source_type'] == 'rtsp':
            source = camera_config['rtsp_url']
            print(f'[CAM] Connecting RTSP: {source}')
            # ⚠️  MUST set env BEFORE VideoCapture() — not after!
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
                'rtsp_transport;tcp'
                '|fflags;nobuffer'
                '|flags;low_delay'
                '|framedrop;1'
                '|analyzeduration;500000'
                '|probesize;500000'
            )
            camera_cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            camera_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            source = int(camera_config['webcam_index'])
            print(f'[CAM] Opening webcam #{source}')
            os.environ.pop('OPENCV_FFMPEG_CAPTURE_OPTIONS', None)
            camera_cap = cv2.VideoCapture(source)
            camera_cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            camera_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera_cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        # Start background frame drainer for ALL sources
        _start_reader(camera_cap)
    return camera_cap

def release_camera():
    global camera_cap, camera_running
    camera_running = False
    _stop_reader()
    _stop_recog_worker()
    if camera_cap and camera_cap.isOpened():
        camera_cap.release()
        camera_cap = None

# ─────────────────────────── Frame generator ───────────────────────────
# Target output FPS sent to browser — high enough to look smooth,
# low enough to never flood the HTTP connection.
_TARGET_FPS   = 25
_FRAME_PERIOD = 1.0 / _TARGET_FPS   # seconds between yielded frames

def gen_frames(mode='preview'):
    """Stream MJPEG to the browser at a steady ~25 FPS.

    Pipeline:
    • _reader_loop drains camera at full speed → _latest_frame  (always fresh)
    • gen_frames wakes at TARGET_FPS, skips if no NEW frame since last send
    • recognition is offloaded to _recognition_worker (async, never blocks stream)
    """
    global camera_running

    with camera_lock:
        get_camera()
        camera_running = True

    if mode == 'recognize':
        _start_recog_worker()

    last_results  = []
    last_sent_seq = -1
    # Lower quality = less bytes per frame = less network backpressure = smoother
    jpeg_quality  = 50 if camera_config['source_type'] == 'rtsp' else 70
    frame_idx     = 0
    RECOG_EVERY   = 8      # send 1 in every 8 frames to recognition worker
                            # (~3 recognition passes/sec at 25 fps)
    next_tick = time.monotonic()

    while camera_running:
        # ── Rate-limit: sleep until next scheduled tick ──────────────
        now = time.monotonic()
        if now < next_tick:
            time.sleep(next_tick - now)
        next_tick = time.monotonic() + _FRAME_PERIOD

        # ── Grab latest frame (no copy if unchanged) ─────────────────
        seq, frame = _grab_latest(timeout=1.5)
        if frame is None:
            continue

        # Skip encoding if we already sent this exact frame
        # (camera hasn't produced a new one yet — prevents duplicate sends)
        if seq == last_sent_seq:
            continue
        last_sent_seq = seq
        frame_idx    += 1

        # ── Async recognition ────────────────────────────────────────
        if mode == 'recognize':
            if frame_idx % RECOG_EVERY == 0:
                try:
                    _recog_input.put_nowait(frame.copy())
                except queue.Full:
                    pass

            try:
                last_results = _recog_output.get_nowait()
            except queue.Empty:
                pass

            for (top, right, bottom, left), name, color in last_results:
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 4, bottom - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        # ── Encode & yield ───────────────────────────────────────────
        _, buf = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')

# ─────────────────────────── Routes ───────────────────────────

@app.route('/')
def index():
    return send_from_directory(str(BASE_DIR), 'index.html')

@app.route('/api/people', methods=['GET'])
def get_people():
    people = load_db()
    safe = [{k: v for k, v in p.items() if k != 'encodings'} for p in people]
    return jsonify(safe)

@app.route('/api/people/<pid>', methods=['GET'])
def get_person(pid):
    people = load_db()
    person = next((p for p in people if p['id'] == pid), None)
    if not person:
        return jsonify({'error': 'Not found'}), 404
    safe = {k: v for k, v in person.items() if k != 'encodings'}
    return jsonify(safe)

@app.route('/api/register', methods=['POST'])
def register():
    name       = request.form.get('name', '').strip()
    emp_id     = request.form.get('emp_id', '').strip()
    department = request.form.get('department', '').strip()

    if not name:
        return jsonify({'error': 'Name is required'}), 400

    # Collect up to 4 photos: photo, photo2, photo3, photo4
    photo_keys = ['photo', 'photo2', 'photo3', 'photo4']
    photo_files = [request.files.get(k) for k in photo_keys if request.files.get(k)]

    if not photo_files:
        return jsonify({'error': 'At least one photo is required'}), 400

    all_encodings = []
    all_photos    = []   # filenames of every saved photo
    pid           = datetime.now().strftime('%Y%m%d%H%M%S%f')
    safe_name     = ''.join(c if c.isalnum() or c in (' ', '_') else '_' for c in name).replace(' ', '_')

    for idx, pf in enumerate(photo_files):
        img_bytes = pf.read()
        nparr     = np.frombuffer(img_bytes, np.uint8)
        img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        locs    = face_recognition.face_locations(img_rgb, model='hog')
        if not locs:
            continue

        enc = face_recognition.face_encodings(img_rgb, locs)[0]
        all_encodings.append(enc.tolist())

        # Save every photo that contains a detected face
        photo_name = f'{safe_name}_{pid}_{idx}.jpg'
        cv2.imwrite(str(FACE_DIR / photo_name), img_bgr)
        all_photos.append(photo_name)
        print(f'[REGISTER] Saved photo {idx+1}/{len(photo_files)}: {photo_name}')

    if not all_encodings:
        return jsonify({'error': 'No face detected in any of the provided photos. Please use clear front-facing photos.'}), 400

    # First saved photo is the primary avatar shown in the UI
    primary_photo = all_photos[0]

    people = load_db()
    people.append({
        'id':         pid,
        'name':       name,
        'emp_id':     emp_id,
        'department': department,
        'photo':      primary_photo,   # avatar shown in the UI
        'photos':     all_photos,      # all saved photos (for re-encoding / audit)
        'registered': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'encodings':  all_encodings,   # one encoding per saved photo
    })
    save_db(people)
    rebuild_encodings()

    return jsonify({'success': True, 'id': pid, 'name': name, 'photos_used': len(all_encodings)})

@app.route('/api/people/<pid>', methods=['DELETE'])
def delete_person(pid):
    people = load_db()
    person = next((p for p in people if p['id'] == pid), None)
    if not person:
        return jsonify({'error': 'Not found'}), 404

    # Remove all saved photos (primary + extra shots)
    all_photos = person.get('photos', [])
    if not all_photos:
        # Fallback for records registered before the multi-photo update
        primary = person.get('photo', '')
        if primary:
            all_photos = [primary]

    for fname in all_photos:
        p = FACE_DIR / fname
        if p.exists():
            p.unlink()

    people = [p for p in people if p['id'] != pid]
    save_db(people)
    rebuild_encodings()
    return jsonify({'success': True})

@app.route('/registered_face/<path:filename>')
def face_photo(filename):
    return send_from_directory(str(FACE_DIR), filename)

@app.route('/face_save/<path:filename>')
def face_save_photo(filename):
    return send_from_directory(str(FACE_SAVE_DIR), filename)

_folder_enc_cache = {}  # {filename: (mtime, encoding_or_none)}

@app.route('/api/folder_scan', methods=['GET'])
def folder_scan():
    """Scan the face_save folder and return info for every image found.
    Stage 1: match by filename against people.json.
    Stage 2: for unmatched files, run face recognition to detect registered people."""
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    people   = load_db()
    person_map = {p['id']: p for p in people}

    # Stage 1 — fast filename lookup
    file_to_person = {}
    for p in people:
        for fname in p.get('photos', [p.get('photo', '')]):
            if fname:
                file_to_person[fname] = p

    entries = []
    if FACE_SAVE_DIR.exists():
        for f in sorted(FACE_SAVE_DIR.iterdir()):
            if f.suffix.lower() not in img_exts:
                continue

            person = file_to_person.get(f.name)
            face_note = ''

            # Stage 2 — face recognition fallback for unmatched files
            if person is None and known_encodings:
                try:
                    mtime = f.stat().st_mtime
                    cached_mtime, enc = _folder_enc_cache.get(f.name, (None, None))
                    if cached_mtime != mtime:
                        img_bgr = cv2.imread(str(f))
                        if img_bgr is not None:
                            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                            locs    = face_recognition.face_locations(img_rgb, model='hog')
                            if locs:
                                encs = face_recognition.face_encodings(img_rgb, locs)
                                enc  = encs[0] if encs else None
                            else:
                                enc  = None
                        else:
                            enc = None
                        _folder_enc_cache[f.name] = (mtime, enc)

                    if enc is not None and len(known_encodings) > 0:
                        distances = face_recognition.face_distance(known_encodings, enc)
                        best_idx  = int(np.argmin(distances))
                        # Generous threshold (0.55) for static photo matching
                        if distances[best_idx] <= 0.55:
                            person    = person_map.get(known_ids[best_idx])
                            face_note = 'face match'
                except Exception as e:
                    print(f'[SCAN] face-recognition error on {f.name}: {e}')

            entries.append({
                'filename':   f.name,
                'url':        f'/face_save/{f.name}',
                'size_kb':    round(f.stat().st_size / 1024, 1),
                'registered': person is not None,
                'face_note':  face_note,           # 'face match' if identified by recognition
                'name':       person.get('name', '')       if person else '',
                'emp_id':     person.get('emp_id', '')     if person else '',
                'department': person.get('department', '') if person else '',
                'person_id':  person.get('id', '')         if person else '',
            })

    return jsonify({'count': len(entries), 'files': entries, 'folder': 'face_save'})

@app.route('/api/recognize_photo', methods=['POST'])
def recognize_photo():
    """Recognize a face in an uploaded photo against the registered database."""
    photo_file = request.files.get('photo')
    if not photo_file:
        return jsonify({'error': 'No photo provided'}), 400

    img_bytes = photo_file.read()
    nparr     = np.frombuffer(img_bytes, np.uint8)
    img_bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({'error': 'Cannot decode image'}), 400

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    locs    = face_recognition.face_locations(img_rgb, model='hog')
    if not locs:
        return jsonify({'error': 'No face detected in the uploaded photo'}), 400

    encs = face_recognition.face_encodings(img_rgb, locs)
    if not encs:
        return jsonify({'error': 'Could not encode face'}), 400

    if not known_encodings:
        return jsonify({'error': 'No registered faces in database'}), 400

    people_db  = load_db()
    person_map = {p['id']: p for p in people_db}

    results = []
    for enc in encs:
        distances = face_recognition.face_distance(known_encodings, enc)
        best_idx  = int(np.argmin(distances))
        best_dist = float(distances[best_idx])
        confidence = round((1 - best_dist) * 100, 1)

        if best_dist <= TOLERANCE:
            pid    = known_ids[best_idx]
            person = person_map.get(pid, {})
            results.append({
                'matched':    True,
                'person_id':  pid,
                'name':       person.get('name', 'Unknown'),
                'emp_id':     person.get('emp_id', ''),
                'department': person.get('department', ''),
                'photo':      person.get('photo', ''),
                'confidence': confidence,
            })
        else:
            results.append({'matched': False, 'confidence': confidence})

    return jsonify({'results': results, 'faces_found': len(locs)})

@app.route('/video_feed/preview')
def video_preview():
    return Response(gen_frames('preview'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/recognize')
def video_recognize():
    return Response(gen_frames('recognize'),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/capture_frame', methods=['POST'])
def capture_frame():
    """Grab a single frame from the server camera and return it as JPEG."""
    with camera_lock:
        get_camera()

    frame = _grab_latest(timeout=3.0)
    if frame is None:
        return jsonify({'error': 'Could not read from camera'}), 500

    ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return jsonify({'error': 'Failed to encode frame'}), 500

    from flask import send_file
    import io
    return send_file(io.BytesIO(buf.tobytes()), mimetype='image/jpeg')

@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    release_camera()
    return jsonify({'success': True})

@app.route('/api/camera/source', methods=['GET'])
def get_camera_source():
    """Return the current camera source configuration."""
    return jsonify(camera_config)

@app.route('/api/camera/scan', methods=['GET'])
def scan_cameras():
    """Probe webcam indexes 0-4 and return those that open successfully."""
    found = []
    for idx in range(5):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            found.append(idx)
            cap.release()
    return jsonify({'cameras': found})

@app.route('/api/camera/source', methods=['POST'])
def set_camera_source():
    """
    Change the camera source at runtime and restart the capture.
    Body (JSON):
      { "source_type": "webcam", "webcam_index": 0 }
      { "source_type": "rtsp",   "rtsp_url": "rtsp://user:pass@host/stream" }
    """
    data = request.get_json(force=True, silent=True) or {}
    source_type = data.get('source_type', '').strip().lower()

    if source_type not in ('webcam', 'rtsp'):
        return jsonify({'error': 'source_type must be "webcam" or "rtsp"'}), 400

    camera_config['source_type'] = source_type

    if source_type == 'webcam':
        try:
            camera_config['webcam_index'] = int(data.get('webcam_index', 0))
        except (TypeError, ValueError):
            return jsonify({'error': 'webcam_index must be an integer'}), 400
    else:
        rtsp_url = data.get('rtsp_url', '').strip()
        if not rtsp_url.startswith('rtsp://'):
            return jsonify({'error': 'rtsp_url must start with rtsp://'}), 400
        camera_config['rtsp_url'] = rtsp_url

    # Release the current camera so get_camera() will reconnect with the new source
    release_camera()
    print(f'[CAM] Source changed to: {camera_config}')
    return jsonify({'success': True, 'config': camera_config})

@app.route('/api/log', methods=['GET'])
def get_log():
    limit = int(request.args.get('limit', 100))
    logs  = load_log()
    return jsonify(logs[:limit])

@app.route('/api/log', methods=['DELETE'])
def clear_log():
    with open(LOG_PATH, 'w') as f:
        json.dump([], f)
    return jsonify({'success': True})

@app.route('/api/log/stream')
def log_stream():
    def event_gen():
        while True:
            try:
                entry = recognition_queue.get(timeout=20)
                yield f"data: {json.dumps(entry)}\n\n"
            except queue.Empty:
                yield ": keep-alive\n\n"
    return Response(event_gen(), mimetype='text/event-stream')

if __name__ == '__main__':
    print('[INFO] Starting server at http://127.0.0.1:5000')
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
