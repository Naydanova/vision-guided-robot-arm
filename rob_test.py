import sys
sys.path.append("/Users/bairma/Documents/Projects/Robot-manipulator/fairino/fairino_sdk/linux")  # for utils
from fairino.Robot import RPC

import time
import math

robot_IP = "192.168.58.2"
robot = RPC(robot_IP)
XMLRPC_PORT = 20003
REALTIME_PORT = 20004

# Move settings
MICRO_DELTA_DEG = 5.0 # delta degrees
MOVE_VEL = 20.0 # velocity in % of max speed, 0 for default speed. has to be float
MOVE_OVL = 20.0 # override blending radius in mm or deg, 0 for no blending, -1 for default blending. has to be float
TOOL_NO = 0 # tool number (TCP), 0 for default tool
USER_NO = 0 # user coordinate system number, 0 for default user coordinate system

def safety_check():
    """
    Perform safety checks before moving the robot.
    """
    print("\n=== SAFETY CHECK ===")
    can_move = True
    
    robot_safe = robot.GetSafetyCode()
    if robot_safe != 0:
        print(f"[BLOCK] SafetyCode = {robot_safe}")
        can_move = False
        err, data = robot.GetRobotErrorCode()
        print(f"RobotErrorCode = {err}, data = {data}")
    else:
        print("[OK] SafetyCode = 0")

    motion_done = robot.GetRobotMotionDone()
    if motion_done[0] != 0:
        print("Error reading motion state:", motion_done[0])
    else:
        if int(motion_done[1]) == 0:
            print("[BLOCK] Robot is busy")
            can_move = False

    pos = robot.GetActualJointPosDegree()
    err, joints = pos
    if err != 0:
        print("[BLOCK] Cannot read joint positions")
        can_move = False
    else:
        print("[OK] Current joints:", joints)

    return joints, can_move


def go_home():
    """
    Home position 

    j1 0.006
    j2 -90.011
    j3 159.116
    j4 -63.64
    j5 130.033
    j6 -130.04

    """
    print("\n=== RETURN TO HOME POSE ===")
    home_point = [0.006, -90.011, 159.116, -63.64, 130.033, -130.04]

    ret = robot.MoveJ(
        home_point,
        TOOL_NO,
        USER_NO,
        vel = MOVE_VEL,
        ovl = MOVE_OVL,
        acc = MOVE_VEL
    )
    if ret != 0:
        print("[ERROR] MoveJ to home failed with code:", ret)
        return ret
    else:
        print("Robot in home pose:", home_point)
        return 0
def main():
    robot_url = f"http://{robot_IP}:{XMLRPC_PORT}"
    print("Connecting to", robot_url)

    try:
        # Basic checks
        sdk_ver = robot.GetSDKVersion()        
        controller_id = robot.GetControllerIP()
        print("Controller IP:", controller_id)
        print("SDK Version:", sdk_ver)

        cur, can_move = safety_check()

        print("\n=== MOVE ===")
        if not can_move:
            print("Move aborted.")
            return

        print("Current joints:", cur)
        target = cur[:]
        target[0] += MICRO_DELTA_DEG

        target = [float(x) for x in target]
        print("[PLAN] Target joints:", target)

        # move = robot.MoveJ(
        #     target,
        #     TOOL_NO,
        #     USER_NO,
        #     vel = MOVE_VEL,
        #     acc = MOVE_VEL,
        #     ovl = MOVE_OVL
        # )

        rob, actual_pose = robot.GetActualTCPPose(0)
        print("Actual pose:", actual_pose)
        target_pose = [
            actual_pose[0],
            actual_pose[1],
            actual_pose[2] + 20,   # только +20 мм по Z
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
        print("Move done:", move)
        print("Joints:", robot.GetActualJointPosDegree())

        # print("[DONE] MoveJ returned:", ret)

        # wait until done
        for _ in range(30):
            md = robot.GetRobotMotionDone()
            if md[0] == 0 and int(md[1]) == 1:
                print("[OK] Motion completed")
                break
            time.sleep(0.2)

        curr_pos = robot.GetActualJointPosDegree()
        print("Current joints:", curr_pos)


        e = go_home()
        if e != 0:
            print("Error returning to home position:", e)
        
    except Exception as e:
        print("XML-RPC failed:", repr(e))
    finally:
        print("Command finished.")



if __name__ == "__main__":
    main()


