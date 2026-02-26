import cv2
import numpy as np
import requests
import time
import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append('fairinoSDK_path')
from fairino.Robot import RPC

ROBOT_IP = os.getenv('ROBOT_IP')
robot = RPC(ROBOT_IP)

# Camera
CAM_STREAM = os.getenv('CAM_STREAM')
CAM_DETS   = os.getenv('CAM_DETS')

# A4 in mm
A4_W, A4_H = 297.0, 210.0

# Robot motion params
Z_SAFE = -140.0
Y_SAFE = 155.0
MOVE_VEL = 20.0
MOVE_OVL = 30.0
TOOL_NO = 1
USER_NO = 1

HOME_POSE = [114.596, -95.52, -113.131,
             -47.504, 82.118, -33.881]

Z_CLEAR = 620.0  # safety height

points = []
matrix = None

detections = []
selected = None  # to be selected
selected_ts = 0.0
SELECT_TTL = 2.0  # safety delay

import inspect
print(inspect.signature(robot.MoveL))
print(robot.MoveL.__doc__)

def order_points_4(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmax(diff)]
    bl = pts[np.argmin(diff)]
    return np.array([tl, bl, br, tr], dtype=np.float32)

def px_to_mm(cx, cy):
    global matrix
    p = np.array([[[cx, cy]]], dtype=np.float32)
    mm = cv2.perspectiveTransform(p, matrix)[0][0]
    return float(mm[0]), float(mm[1])

def fetch_dets():
    try:
        r = requests.get(CAM_DETS, timeout=0.5)
        r.raise_for_status()
        js = r.json()
        dets = js.get("detections", [])
        if not isinstance(dets, list):
            print("WARN: detections is not a list:", type(dets), js)
            return []
        return dets
    except Exception as e:
        if int(time.time() * 2) % 4 == 0:
            print("WARN fetch_dets:", repr(e))
        return []

def pick_best(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: d.get("conf", 0.0))

def move_robot_xy(x_mm, y_mm):
    global Z_SAFE, MOVE_VEL, MOVE_OVL, TOOL_NO, USER_NO, Y_SAFE
    actual_pose = robot.GetActualTCPPose(0)[1]
    print("Actual pose: ", actual_pose)

    rx, ry, rz = actual_pose[3], actual_pose[4], actual_pose[5]
    z = actual_pose[1]   # safety height

    # target_pose = [x_mm, y_mm, z, rx, ry, rz]
    target_pose = [float(x_mm), float(Y_SAFE), float(y_mm), float(rx), float(ry), float(rz)]
    print("GO TO:", target_pose)
    # print("DEBUG target_pose type:", 
    #     type(target_pose), "len:", (len(target_pose) 
    #     if hasattr(target_pose, "__len__") else None)
    # )
    # print("DEBUG target_pose:", target_pose)
    # assert isinstance(target_pose, (list, tuple)) and len(target_pose) == 6, f"target_pose must be list/tuple len=6, got {type(target_pose)} {target_pose}"

    move = robot.MoveL(
        target_pose,
        TOOL_NO,
        USER_NO,  # wobj we created on Fairino GUI
        joint_pos=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # test
        vel=MOVE_VEL,
        acc=MOVE_VEL,
        ovl=MOVE_OVL       
    )
    print("MoveL return:", move)

def on_mouse(event, x, y, flags, param):
    global points, matrix
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # for temporal calibration
    if matrix is None:
        points.append((x, y))
        print("A4 point", len(points), ":", (x, y))
        if len(points) == 4:
            img_pts = order_points_4(points)
            real_pts = np.array([[0,0], [0,A4_H], [A4_W,A4_H], [A4_W,0]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(img_pts, real_pts)
            print("[CALIB] matrix ready")            
        return

def print_pose_delta(tag=""):
    p = robot.GetActualTCPPose(0)[1]
    print(tag, "pose:", [round(x,3) for x in p])
    return [float(x) for x in p]

def delta(a, b):
    return [round(b[i]-a[i], 3) for i in range(6)]

def nudge(dx=10.0, dy=0.0, dz=0.0):
    p0 = print_pose_delta("BEFORE")
    tgt = [p0[0]+dx, p0[1]+dy, p0[2]+dz, p0[3], p0[4], p0[5]]
    print("TARGET:", [round(x,3) for x in tgt])
    robot.MoveL(tgt, TOOL_NO, USER_NO, vel=5.0, acc=5.0, ovl=5.0)
    p1 = print_pose_delta("AFTER ")
    print("DELTA :", delta(p0, p1))

def go_home():
    pose = robot.GetActualTCPPose(0)[1]
    cur = [float(v) for v in pose]
    home = [float(v) for v in HOME_POSE]

    print("GO_HOME from:", cur)
    print("GO_HOME to  :", home)

    # 1) up
    up = [cur[0], cur[1], max(cur[2], Z_CLEAR), cur[3], cur[4], cur[5]]
    print("GO_HOME step1 (up):", up)
    ret1 = robot.MoveL(up, TOOL_NO, USER_NO, vel=10.0, acc=10.0, ovl=10.0)
    print("GO_HOME ret1:", ret1)

    # 2) go-to-target
    mid = [home[0], home[1], up[2], home[3], home[4], home[5]]
    print("GO_HOME step2 (xy):", mid)
    ret2 = robot.MoveL(mid, TOOL_NO, USER_NO, vel=10.0, acc=10.0, ovl=10.0)
    print("GO_HOME ret2:", ret2)

    # 3) down
    print("GO_HOME step3 (down):", home)
    ret3 = robot.MoveL(home, TOOL_NO, USER_NO, vel=5.0, acc=5.0, ovl=5.0)
    print("GO_HOME ret3:", ret3)

    return ret1, ret2, ret3

def main():
    global detections, selected

    cap = cv2.VideoCapture(CAM_STREAM)
    if not cap.isOpened():
        raise RuntimeError("Can't open CAM_STREAM. Check URL.")

    win_name = "cam"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    last_pull = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        now = time.time()
        if now - last_pull > 0.15:
            dets = fetch_dets()
            if dets:  # data update
                detections = dets
                selected = pick_best(detections)
                selected_ts = now
            else:
                # if data is outdated
                if selected is not None and (now - selected_ts) > SELECT_TTL:
                    selected = None
                    detections = []
            last_pull = now

        # draw detections
        for d in detections:
            x1,y1,x2,y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

        # highlight selected + show center
        if selected is not None:
            x1,y1,x2,y2 = selected["x1"], selected["y1"], selected["x2"], selected["y2"]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            cv2.circle(frame, (cx,cy), 6, (0,0,255), -1)
            cv2.putText(frame, f"px: {cx},{cy}", (cx+10, cy-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            if matrix is not None:
                mmx, mmy = px_to_mm(cx, cy)
                cv2.putText(frame, f"mm: {mmx:.1f},{mmy:.1f}", (cx+10, cy+15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # draw A4 points
        for i, p in enumerate(points):
            cv2.circle(frame, p, 6, (255,0,0), -1)
            cv2.putText(frame, str(i+1), (p[0]+8, p[1]-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        cv2.putText(frame, "Click 4 A4 corners. Then press [g]=go, [r]=reset, [ESC]=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        if key == ord('r'):
            points.clear()
            # reset calib
            globals()["matrix"] = None
            print("[RESET]")
        if key == ord('g'):
            if matrix is None or selected is None or (time.time() - selected_ts) > SELECT_TTL:
                print("No calibrarion or target")
                continue
            x1,y1,x2,y2 = selected["x1"], selected["y1"], selected["x2"], selected["y2"]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            mmx, mmy = px_to_mm(cx, cy)
            # mmx = A4_W - mmx
            move_robot_xy(mmx, mmy)
        
        if key == ord('h'):
            go_home()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()