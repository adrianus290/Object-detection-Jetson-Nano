import cv2
import numpy as np
import subprocess
import sys
import threading
import time
import onnxruntime as ort

# ==============================================================================
# 1. INISIALISASI & KONFIGURASI AMAN JETSON GPIO
# ==============================================================================
try:
    import Jetson.GPIO as GPIO
    JETSON_GPIO_AVAILABLE = True
except ImportError:
    JETSON_GPIO_AVAILABLE = False
    print(" WARNING: Library Jetson.GPIO tidak ditemukan.")

LED_MERAH = 18    # Pin Fisik 11 (STOP)
LED_KUNING = 12   # Pin Fisik 13 (SLOW)
LED_HIJAU =  16   # Pin Fisik 16 (GO)

LED_ACTIVE_LOW = False

def _led_level(nyala):
    if LED_ACTIVE_LOW:
        return GPIO.LOW if nyala else GPIO.HIGH
    return GPIO.HIGH if nyala else GPIO.LOW

if JETSON_GPIO_AVAILABLE:
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    for pin in [LED_MERAH, LED_KUNING, LED_HIJAU]:
        try:
            GPIO.setup(pin, GPIO.OUT, initial=_led_level(False))
        except Exception as e:
            print(f"Gagal setup pin {pin}: {e}")

def set_leds(merah, kuning, hijau):
    if not JETSON_GPIO_AVAILABLE:
        return
    try:
        GPIO.output(LED_MERAH, _led_level(bool(merah)))
        GPIO.output(LED_KUNING, _led_level(bool(kuning)))
        GPIO.output(LED_HIJAU, _led_level(bool(hijau)))
    except Exception as e:
        print(f"Error setting LED: {e}")

set_leds(False, False, False)

# ==============================================================================
# 2. INISIALISASI ONNX RUNTIME (YOLO11)
# ==============================================================================
MODEL_ONNX_PATH = 'best1.onnx'
CONFIDENCE_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

CLASS_NAMES = ['GO', 'SLOW', 'STOP']

print(" Memuat model YOLO11 ONNX menggunakan ONNXRuntime...")
try:
    session = ort.InferenceSession(MODEL_ONNX_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(" Model YOLO11 ONNX berhasil dimuat!")
except Exception as e:
    print(f" Error memuat model ONNX: {e}")
    exit()

try:
    import ast
    metadata_names = session.get_modelmeta().custom_metadata_map.get('names')
    if metadata_names:
        parsed_names = ast.literal_eval(metadata_names)
        CLASS_NAMES = [parsed_names[i] for i in sorted(parsed_names)]
        print(f" Nama kelas dari model: {CLASS_NAMES}")
except Exception as e:
    print(f"Gagal membaca nama kelas dari model, pakai default: {e}")

# ==============================================================================
# 3. OPTIMIZED CAMERA STREAM (LATEST FRAME BUFFER)
# ==============================================================================
class CameraStream:
    def __init__(self, command, width, height):
        self.width = width
        self.height = height
        self.frame_bytes = width * height * 2
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.frame = None
        self.running = True
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            raw_image = self.process.stdout.read(self.frame_bytes)
            if len(raw_image) == self.frame_bytes:
                frame_yuv = np.frombuffer(raw_image, dtype=np.uint8).reshape((self.height, self.width, 2))
                frame_bgr = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR_YUY2)

                with self.lock:
                    self.frame = frame_bgr

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.process.terminate()

# ==============================================================================
# 4. FUNGSI INFERENCE YOLO11
# ==============================================================================
def detect_objects_yolo11(frame):
    h, w, _ = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (INPUT_WIDTH, INPUT_HEIGHT),
                                 swapRB=True, crop=False)

    outputs = session.run([output_name], {input_name: blob})[0]
    outputs = outputs[0].T   # Shape -> [8400, 4 + num_classes]

    classes_scores = outputs[:, 4:]
    class_ids = np.argmax(classes_scores, axis=1)
    max_scores = classes_scores[np.arange(len(class_ids)), class_ids]
    mask = max_scores >= CONFIDENCE_THRESHOLD

    boxes_xywh = outputs[mask, :4]
    scores = max_scores[mask]
    class_ids = class_ids[mask]

    x_factor = w / INPUT_WIDTH
    y_factor = h / INPUT_HEIGHT

    cx, cy, pw, ph = boxes_xywh.T
    left = ((cx - pw / 2) * x_factor).astype(int)
    top = ((cy - ph / 2) * y_factor).astype(int)
    box_w = (pw * x_factor).astype(int)
    box_h = (ph * y_factor).astype(int)

    boxes = np.stack([left, top, box_w, box_h], axis=1).tolist()
    scores = scores.tolist()
    class_ids = class_ids.tolist()

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
    if isinstance(indices, tuple):
        indices = indices[0]
    if indices is not None:
        indices = np.array(indices).flatten()
    else:
        indices = []

    detected_results = []
    for i in indices:
        detected_results.append({
            'box': boxes[int(i)],
            'score': scores[int(i)],
            'class_id': class_ids[int(i)]
        })

    return detected_results

# ==============================================================================
# 5. THREAD DETEKSI (MURNI REAL-TIME, TANPA DELAY)
# ==============================================================================
detector_lock = threading.Lock()
latest_results = []
latest_inference_ms = 0.0
detector_running = True

def detection_worker():
    global latest_results, latest_inference_ms
    while detector_running:
        frame = cam.read()
        if frame is None:
            continue

        start = time.time()
        try:
            res = detect_objects_yolo11(frame)
        except Exception as e:
            res = []  # Jika error saat inferensi, pastikan langsung kosong
        elapsed_ms = (time.time() - start) * 1000.0

        with detector_lock:
            # Mengganti langsung hasil deteksi secara instan (jika kosong, jadi [])
            latest_results = res
            latest_inference_ms = elapsed_ms

# ==============================================================================
# 6. MAIN LOOP
# ==============================================================================
WIDTH = 768
HEIGHT = 640
FPS = 30

gst_command = [
    "gst-launch-1.0", "-q",
    "nvarguscamerasrc", "sensor-mode=2", "!",
    f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, framerate={FPS}/1", "!",
    "nvvidconv", "flip-method=2", "!",
    "video/x-raw, format=YUY2", "!",
    "queue", "leaky=downstream", "max-size-buffers=1", "!",
    "fdsink"
]

print("Membuka stream kamera yang dioptimalkan...")
cam = CameraStream(gst_command, WIDTH, HEIGHT)
time.sleep(2)

detector_thread = threading.Thread(target=detection_worker, daemon=True)
detector_thread.start()

print("[SUKSES] Sistem berjalan *real-time*! Tekan 'q' untuk keluar\n")

prev_state = None
fps_start = time.time()
fps_frames = 0
display_fps = 0.0

try:
    while True:
        frame = cam.read()
        if frame is None:
            continue

        fps_frames += 1
        now = time.time()
        if now - fps_start >= 1.0:
            display_fps = fps_frames / (now - fps_start)
            fps_frames = 0
            fps_start = now

        # Ambil hasil deteksi terbaru secara langsung (tanpa timeout)
        with detector_lock:
            results = list(latest_results)
            inference_ms = latest_inference_ms

        detected_stop = False
        detected_slow = False
        detected_go = False

        # Render bounding box hanya jika results ada isinya
        for det in results:
            box = det['box']
            score = det['score']
            cls_id = det['class_id']

            label_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Class {cls_id}"
            label_key = label_name.upper()

            if label_key == 'STOP':
                detected_stop = True
            elif label_key == 'SLOW':
                detected_slow = True
            elif label_key == 'GO':
                detected_go = True

            left, top, width, height = box
            cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Logika LED
        if detected_stop:
            state = 'stop'
            status_text = "PREDIKSI: STOP -> LED MERAH ON"
            text_color = (0, 0, 255)
            set_leds(merah=True, kuning=False, hijau=False)
        elif detected_slow:
            state = 'slow'
            status_text = "PREDIKSI: SLOW -> LED KUNING ON"
            text_color = (0, 255, 255)
            set_leds(merah=False, kuning=True, hijau=False)
        elif detected_go:
            state = 'go'
            status_text = "PREDIKSI: GO -> LED HIJAU ON"
            text_color = (0, 255, 0)
            set_leds(merah=False, kuning=False, hijau=True)
        else:
            state = 'none'
            status_text = "STATUS: TIDAK ADA RAMBU (All LEDs OFF)"
            text_color = (255, 255, 255)
            set_leds(merah=False, kuning=False, hijau=False)

        if state != prev_state:
            print(f"[{time.strftime('%H:%M:%S')}] LED: {state.upper()}")
            prev_state = state

        # Tampilkan teks status & performa
        cv2.rectangle(frame, (20, 20), (620, 80), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        cv2.putText(frame, f"FPS: {display_fps:.1f} | AI: {inference_ms:.0f} ms",
                    (30, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Jetson Nano - Real-Time Traffic Sign", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nMematikan aplikasi...")
            break

except KeyboardInterrupt:
    print("\nProgram dihentikan oleh pengguna.")

finally:
    detector_running = False
    cam.release()
    cv2.destroyAllWindows()
    set_leds(False, False, False)
    if JETSON_GPIO_AVAILABLE:
        try:
            GPIO.cleanup()
        except:
            pass
    print("Sistem berhasil dibersihkan.")
