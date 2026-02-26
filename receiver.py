# receiver.py
import os
from dotenv import load_dotenv
load_dotenv()
import cv2
import time
import requests
import sys
fairinoSDK_path = os.getenv('FAIRINO_SDK_PATH')
sys.path.append('fairinoSDK_path')
from fairino.Robot import RPC

robot_IP = os.getenv('ROBOT_IP')
robot = RPC(robot_IP)

# Camera
CAM_STREAM = os.getenv('CAM_STREAM')
CAM_DETS   = os.getenv('CAM_DETS')

# Coords
Z_SAFE  = 120.0
Z_GRASP =  25.0

detections = []       # list of dicts: x1,y1,x2,y2,cls,conf
selected_idx = None   # bbox index

def inside_bbox(x, y, d):
    return d["x1"] <= x <= d["x2"] and d["y1"] <= y <= d["y2"]

def on_mouse(event, x, y, flags, param):
    global selected_idx
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    hits = [i for i,d in enumerate(detections) if inside_bbox(x,y,d)]
    if not hits:
        selected_idx = None
        return
    selected_idx = max(hits, key=lambda i: detections[i]["conf"])
    print("Selected idx:", selected_idx, "det:", detections[selected_idx])

def fetch_detections():
    try:
        r = requests.get(CAM_DETS, timeout=0.2)
        r.raise_for_status()
        data = r.json()
        return data.get("detections", [])
    except Exception:
        return []

def robot_go_xy(x_mm, y_mm):
    MICRO_DELTA_DEG = 5.0 # delta degrees
    MOVE_VEL = 20.0 # velocity in % of max speed, 0 for default speed. has to be float
    MOVE_OVL = 20.0 # override blending radius in mm or deg, 0 for no blending, -1 for default blending. has to be float
    TOOL_NO = 0 # tool number (TCP), 0 for default tool
    USER_NO = 0 # user coordinate system number, 0 for default user coordinate system

    target_pose = [
        actual_pose[0],
        actual_pose[1],
        actual_pose[2] + 20,   # only +20 мм on Z
        actual_pose[3],
        actual_pose[4],
        actual_pose[5]
    ]
    move = robot.MoveL(
        target_pose,
        TOOL_NO,
        USER_NO,
        vel = MOVE_VEL,
        acc = MOVE_VEL,
        ovl = MOVE_OVL
    )
    print(f"[ROBOT] GO XY mm: {x_mm:.1f}, {y_mm:.1f}")
    # robot.MoveL([...]) или robot.MoveJ([...])

def main():
    global detections, selected_idx

    cap = cv2.VideoCapture(CAM_STREAM)
    if not cap.isOpened():
        raise RuntimeError("Can't open CAM_STREAM. Check URL.")

    win_name = "cam"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name,on_mouse)

    last_pull = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        now = time.time()
        if now - last_pull > 0.1:
            detections = fetch_detections()
            last_pull = now

        # making bbox
        for i, d in enumerate(detections):
            x1,y1,x2,y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            is_sel = (i == selected_idx)
            color = (0,0,255) if is_sel else (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

        cv2.putText(frame, "Click bbox to select. Press [g]=go, [c]=clear, [ESC]=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key == ord('c'):
            selected_idx = None
        if key == ord('g'):
            if selected_idx is None:
                print("No selection")
                continue
            d = detections[selected_idx]
            cx = (d["x1"] + d["x2"]) // 2
            cy = (d["y1"] + d["y2"]) // 2

            raise RuntimeError("Add pixel->mm maping here")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# # Move settings
# MICRO_DELTA_DEG = 40.0 # delta degrees
# MOVE_VEL = 20.0 # velocity in % of max speed, 0 for default speed. has to be float
# MOVE_OVL = 20.0 # override blending radius in mm or deg, 0 for no blending, -1 for default blending. has to be float
# TOOL_NO = 0 # tool number (TCP), 0 for default tool
# USER_NO = 0 # user coordinate system number, 0 for default user coordinate system

# app = FastAPI()

# IMG_PTS = np.array([
#     [100, 100],
#     [1180, 100],
#     [1180, 620],
#     [100, 620],
# ], dtype=np.float32)

# TABLE_PTS = np.array([
#     [0,   0],
#     [300, 0],
#     [300, 200],
#     [0,   200],
# ], dtype=np.float32)

# H, _ = cv2.findHomography(IMG_PTS, TABLE_PTS)  # 3x3

# # Z_UP = 120   # мм
# # Z_DOWN = 30  # мм (подбери)

# class Target(BaseModel):
#     u: float
#     v: float

# def uv_to_xy_mm(u, v):
#     p = np.array([[[u, v]]], dtype=np.float32)        # shape (1,1,2)
#     xy = cv2.perspectiveTransform(p, H)[0,0]          # (X,Y)
#     return float(xy[0]), float(xy[1])

# @app.post("/target")
# def target(t: Target):
#     X, Y = uv_to_xy_mm(t.u, t.v)
#     Z_up = 120

#     print(f"Computed XY: {X:.2f}, {Y:.2f}")

#     if not (0 <= X <= 300 and 0 <= Y <= 200):
#         print("Out of bounds")
#         return {"ok": False}

#     try:
#         pose = [X, Y, Z_up, 180, 0, 180]

#         print("Sending MoveL:", pose)

#         ret = robot.MoveL(
#             pose,
#             vel=MOVE_VEL,
#             acc=MOVE_VEL,
#             ovl=0,  # без блендинга пока
#             tool=TOOL_NO,
#             user=USER_NO
#         )

#         print("Robot response:", ret)

#     except Exception as e:
#         print("Move failed:", e)
#         return {"ok": False}

#     return {"X_mm": X, "Y_mm": Y, "Z_up": Z_up}


