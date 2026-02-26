# camera_server_stdlib.py
import os
from dotenv import load_dotenv
load_dotenv()
import json
import time
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import importlib.util

HOST = os.getenv('CAM_HOST')
PORT = os.getenv('CAM_PORT')

# ---- Paths / settings ----
MODEL_DIR = os.getenv('CMODEL_DIR')
TFLITE_FILE = os.getenv('TFLITE_FILE')
LABELS_FILE = os.getenv('LABELS_FILE')

THRESHOLD = 0.60
NMS_IOU = 0.45
DET_PERIOD = 0.15   # сек, ~6-7 FPS детекций
JPEG_QUALITY = 80

CAM_DEVICE = os.getenv('CAM_DEVICE')
CAM_W, CAM_H = 640, 480

# ---- Shared state for HTTP ----
latest_jpeg = None
latest_dets = []
lock = threading.Lock()

# ---- TFLite global ----
interpreter = None
input_details = None
output_details = None
input_index = None
output_index = None
labels = None


def v4l2_gstreamer_pipeline(src=CAM_DEVICE, display_width=640, display_height=480):
    return (
        "v4l2src device=%s ! "
        "decodebin ! "
        "imxvideoconvert_g2d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert n-threads=3 ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink "
        % (src, display_width, display_height)
    )


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    im_resized = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_padded = cv2.copyMakeBorder(
        im_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return im_padded, r, (dw, dh)


def adapt_output_from_tflite(output: np.ndarray) -> np.ndarray:
    # (1, 9, 8400) -> (8400, 9)
    # (1, 8400, 9) -> (8400, 9)
    if output.ndim == 3 and output.shape[0] == 1:
        if output.shape[1] == 9:
            return output[0].T
        if output.shape[2] == 9:
            return output[0]
    raise ValueError(f"Unexpected tflite output shape: {output.shape}")


def nms(boxes, scores, iou_threshold=0.45): #threshold to be updated
    if len(boxes) == 0:
        return np.array([], dtype=int)

    boxes = boxes.astype(float)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=int)


def init_detector():
    global interpreter, input_details, output_details, input_index, output_index, labels

    # labels
    labels_path = os.path.join(MODEL_DIR, LABELS_FILE)
    with open(labels_path, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    if labels and labels[0] == "???":
        labels = labels[1:]

    # interpreter
    pkg = importlib.util.find_spec("tflite_runtime")
    if pkg:
        from tflite_runtime.interpreter import Interpreter
    else:
        from tensorflow.lite.python.interpreter import Interpreter  # just in case

    model_path = os.path.join(MODEL_DIR, TFLITE_FILE)
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    output_index = output_details[0]["index"]

    print("MODEL:", model_path)
    print("INPUT dtype:", input_details[0]["dtype"])
    print("INPUT shape:", input_details[0]["shape"])
    print("INPUT quant:", input_details[0]["quantization"])
    print("OUTPUT dtype:", output_details[0]["dtype"])
    print("OUTPUT shape:", output_details[0]["shape"])
    print("OUTPUT quant:", output_details[0]["quantization"])


def detect(frame) -> list:
    # returns JSON-ready list of dicts
    in_info = input_details[0]
    in_shape = in_info["shape"]
    in_dtype = in_info["dtype"]
    in_scale, in_zero = in_info["quantization"]

    if len(in_shape) != 4 or in_shape[0] != 1:
        raise RuntimeError(f"Unexpected input shape: {in_shape}")

    if in_shape[1] == 3:
        layout = "NCHW"
        in_h, in_w = int(in_shape[2]), int(in_shape[3])
    else:
        layout = "NHWC"
        in_h, in_w = int(in_shape[1]), int(in_shape[2])

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_lb, r, (dw, dh) = letterbox(img_rgb, (in_h, in_w))
    img = img_lb.astype(np.float32) / 255.0

    # quantize if needed
    if in_dtype == np.float32:
        img_proc = img
    elif in_dtype in (np.uint8, np.int8):
        if in_scale == 0:
            raise RuntimeError("Input quant scale == 0")
        q = img / in_scale + in_zero
        if in_dtype == np.uint8:
            img_proc = np.clip(np.round(q), 0, 255).astype(np.uint8)
        else:
            img_proc = np.clip(np.round(q), -128, 127).astype(np.int8)
    else:
        raise RuntimeError(f"Unexpected input dtype: {in_dtype}")

    if layout == "NHWC":
        img_input = np.expand_dims(img_proc, axis=0)
    else:
        img_input = np.expand_dims(np.transpose(img_proc, (2, 0, 1)), axis=0)

    # inference
    interpreter.set_tensor(input_index, img_input)
    interpreter.invoke()
    raw_output = interpreter.get_tensor(output_index)

    # dequant output
    out_scale, out_zero = output_details[0]["quantization"]
    out_dtype = output_details[0]["dtype"]
    if out_dtype in (np.uint8, np.int8):
        if out_scale == 0:
            raise RuntimeError("Output quant scale == 0")
        out = (raw_output.astype(np.float32) - out_zero) * out_scale
    else:
        out = raw_output.astype(np.float32)

    pred = adapt_output_from_tflite(out)  # (N,9)
    if pred.shape[1] != 9:
        raise RuntimeError(f"Expected (N,9), got {pred.shape}")

    boxes_xywh = pred[:, :4]
    class_scores = pred[:, 4:]  # 5 classes

    cls_ids = np.argmax(class_scores, axis=1)
    conf = class_scores[np.arange(class_scores.shape[0]), cls_ids]

    mask = conf >= THRESHOLD
    boxes_xywh = boxes_xywh[mask]
    cls_ids = cls_ids[mask]
    conf = conf[mask]

    if boxes_xywh.shape[0] == 0:
        return []

    h_out, w_out, _ = frame.shape

    cx = boxes_xywh[:, 0] * in_w
    cy = boxes_xywh[:, 1] * in_h
    bw = boxes_xywh[:, 2] * in_w
    bh = boxes_xywh[:, 3] * in_h

    cx -= dw
    cy -= dh

    cx /= r
    cy /= r
    bw /= r
    bh /= r

    x1 = np.clip(cx - bw / 2, 0, w_out - 1).astype(int)
    y1 = np.clip(cy - bh / 2, 0, h_out - 1).astype(int)
    x2 = np.clip(cx + bw / 2, 0, w_out - 1).astype(int)
    y2 = np.clip(cy + bh / 2, 0, h_out - 1).astype(int)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms(boxes_xyxy, conf, iou_threshold=NMS_IOU)
    boxes_xyxy = boxes_xyxy[keep]
    cls_ids = cls_ids[keep]
    conf = conf[keep]

    dets = []
    for (x1, y1, x2, y2), cls_id, score in zip(boxes_xyxy, cls_ids, conf):
        dets.append({
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "cls": int(cls_id),
            "name": labels[cls_id] if labels and 0 <= int(cls_id) < len(labels) else str(int(cls_id)),
            "conf": float(score),
        })
    return dets


def capture_and_detect_loop():
    global latest_jpeg, latest_dets

    init_detector()

    cap = cv2.VideoCapture(v4l2_gstreamer_pipeline(CAM_DEVICE, CAM_W, CAM_H), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    last_det_t = 0.0
    dets_cache = []

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        now = time.time()
        if now - last_det_t >= DET_PERIOD:
            try:
                dets_cache = detect(frame)
            except Exception as e:
                print("[DETECT ERROR]", e)
                dets_cache = []
            last_det_t = now

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            continue

        with lock:
            latest_jpeg = jpg.tobytes()
            latest_dets = dets_cache


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/detections"):
            with lock:
                dets = list(latest_dets)
            body = json.dumps({"ts": time.time(), "detections": dets}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpg = latest_jpeg
                    if jpg is None:
                        time.sleep(0.02)
                        continue
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("utf-8"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except BrokenPipeError:
                return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def main():
    t = threading.Thread(target=capture_and_detect_loop, daemon=True)
    t.start()

    srv = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Serving on http://{HOST}:{PORT}  endpoints: /stream  /detections")
    srv.serve_forever()


if __name__ == "__main__":
    main()