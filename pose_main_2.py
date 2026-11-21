import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import numpy.typing as npt
from pythonosc.udp_client import SimpleUDPClient

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("mediapipe is required. Please install dependencies from requirements.txt") from e

# Mediapipe setup
mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]

# Landmarks of interest for arms/hands
LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST.value      # 15
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST.value    # 16
LEFT_ELBOW = mp_pose.PoseLandmark.LEFT_ELBOW.value      # 13
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW.value    # 14
LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value  # 11
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value # 12

# Selected landmarks for each hand (2-4 points as requested)
LEFT_ARM_LANDMARKS = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]   # 3 points for left arm
RIGHT_ARM_LANDMARKS = [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST] # 3 points for right arm

# Use wrist for dynamics calculation
SELECTED_JOINT_FOR_DYNAMICS_LEFT = LEFT_WRIST
SELECTED_JOINT_FOR_DYNAMICS_RIGHT = RIGHT_WRIST

# Joint names for debug output
JOINT_NAMES = {
    LEFT_SHOULDER: "Left_Shoulder",
    LEFT_ELBOW: "Left_Elbow", 
    LEFT_WRIST: "Left_Wrist",
    RIGHT_SHOULDER: "Right_Shoulder",
    RIGHT_ELBOW: "Right_Elbow",
    RIGHT_WRIST: "Right_Wrist"
}

Vec3 = npt.NDArray[np.float64]


@dataclass
class ArmState:
    p: Optional[Vec3] = None
    v: Optional[Vec3] = None
    t: Optional[float] = None


def flatten_landmarks(landmarks, indices):
    """
    Flatten selected landmark coordinates [x, y, z] into a single list.
    landmarks: list of NormalizedLandmark
    indices: list of landmark indices to include
    Returns: list[float]
    """
    out: List[float] = []
    for i in indices:
        lm = landmarks[i]
        out.extend([lm.x, lm.y, lm.z])
    return out


def main():
    parser = argparse.ArgumentParser(description="Pose OSC sender: Mediapipe Pose -> OSC for Max/MSP")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Destination IP address for OSC")
    parser.add_argument("--port_positions", type=int, default=9001, help="OSC UDP port for positions list")
    parser.add_argument("--port_dynamics", type=int, default=9002, help="OSC UDP port for velocity/acceleration list")
    parser.add_argument("--camera", type=int, default=0, help="Camera index for cv2.VideoCapture")
    parser.add_argument("--min_detection_confidence", type=float, default=0.6)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
    args = parser.parse_args()

    client_positions = SimpleUDPClient(args.ip, args.port_positions)
    client_dynamics = SimpleUDPClient(args.ip, args.port_dynamics)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open camera index {args.camera}")

    # State for dynamics per arm ('Left'/'Right')
    prev_state: Dict[str, ArmState] = {
        'Left': ArmState(),
        'Right': ArmState(),
    }

    # Constant structure lengths for OSC parsing in Max
    per_arm_positions_len = 3 * 3  # 3 joints * (x,y,z) per arm
    per_arm_dynamics_len = 6       # (vx,vy,vz, ax,ay,az) per arm

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    ) as pose:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    # Wait a bit and try again
                    time.sleep(0.005)
                    continue

                # Convert BGR to RGB for Mediapipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = pose.process(rgb)
                t_now = time.perf_counter()
                
                # Draw pose landmarks on the frame for visualization
                rgb.flags.writeable = True
                annotated_image = rgb.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                    )
                
                # Display the annotated frame
                cv2.imshow('Pose Detection', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

                # Prepare containers for left/right arms
                positions: Dict[str, Optional[List[float]]] = {
                    'Left': None,
                    'Right': None,
                }
                dynamics: Dict[str, Optional[List[float]]] = {
                    'Left': None,
                    'Right': None,
                }

                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    
                    # Check if key landmarks are visible (confidence > threshold)
                    visibility_threshold = 0.5
                    
                    # Process left arm
                    left_visible = all(lms[idx].visibility > visibility_threshold for idx in LEFT_ARM_LANDMARKS)
                    if left_visible:
                        # Build positions list for left arm
                        pos_list = flatten_landmarks(lms, LEFT_ARM_LANDMARKS)
                        positions['Left'] = pos_list
                        
                        # Debug output: print joint coordinates
                        print(f"\nLeft Arm:")
                        for idx in LEFT_ARM_LANDMARKS:
                            joint = lms[idx]
                            joint_name = JOINT_NAMES[idx]
                            print(f"  {joint_name}: x={joint.x:.3f}, y={joint.y:.3f}, z={joint.z:.3f}")
                        
                        # Dynamics based on left wrist
                        joint = lms[SELECTED_JOINT_FOR_DYNAMICS_LEFT]
                        p = np.array([joint.x, joint.y, joint.z], dtype=np.float64)
                        
                        prev = prev_state['Left']
                        if prev.p is None:
                            # Initialize
                            v = np.zeros(3, dtype=np.float64)
                            a = np.zeros(3, dtype=np.float64)
                        else:
                            prev_t = prev.t if prev.t is not None else t_now
                            dt = max(t_now - prev_t, 1e-6)
                            v = (p - prev.p) / dt
                            if prev.v is None:
                                a = np.zeros(3, dtype=np.float64)
                            else:
                                a = (v - prev.v) / dt
                        
                        # Update state
                        prev_state['Left'].p = p
                        prev_state['Left'].v = v
                        prev_state['Left'].t = t_now
                        
                        dynamics['Left'] = [
                            float(v[0]), float(v[1]), float(v[2]),
                            float(a[0]), float(a[1]), float(a[2])
                        ]
                    
                    # Process right arm
                    right_visible = all(lms[idx].visibility > visibility_threshold for idx in RIGHT_ARM_LANDMARKS)
                    if right_visible:
                        # Build positions list for right arm
                        pos_list = flatten_landmarks(lms, RIGHT_ARM_LANDMARKS)
                        positions['Right'] = pos_list
                        
                        # Debug output: print joint coordinates
                        print(f"\nRight Arm:")
                        for idx in RIGHT_ARM_LANDMARKS:
                            joint = lms[idx]
                            joint_name = JOINT_NAMES[idx]
                            print(f"  {joint_name}: x={joint.x:.3f}, y={joint.y:.3f}, z={joint.z:.3f}")
                        
                        # Dynamics based on right wrist
                        joint = lms[SELECTED_JOINT_FOR_DYNAMICS_RIGHT]
                        p = np.array([joint.x, joint.y, joint.z], dtype=np.float64)
                        
                        prev = prev_state['Right']
                        if prev.p is None:
                            # Initialize
                            v = np.zeros(3, dtype=np.float64)
                            a = np.zeros(3, dtype=np.float64)
                        else:
                            prev_t = prev.t if prev.t is not None else t_now
                            dt = max(t_now - prev_t, 1e-6)
                            v = (p - prev.p) / dt
                            if prev.v is None:
                                a = np.zeros(3, dtype=np.float64)
                            else:
                                a = (v - prev.v) / dt
                        
                        # Update state
                        prev_state['Right'].p = p
                        prev_state['Right'].v = v
                        prev_state['Right'].t = t_now
                        
                        dynamics['Right'] = [
                            float(v[0]), float(v[1]), float(v[2]),
                            float(a[0]), float(a[1]), float(a[2])
                        ]
                else:
                    # No pose detected: reset states to avoid large spikes when reappearing
                    for k in list(prev_state.keys()):
                        prev_state[k] = ArmState()

                # Compose unified lists (Left then Right), each preceded by a present flag for robustness in Max
                # Positions list layout per arm: [present, shoulder(3), elbow(3), wrist(3)]
                positions_msg: List[float] = []
                for label in ('Left', 'Right'):
                    if positions[label] is None:
                        positions_msg.append(0.0)
                        positions_msg.extend([0.0] * per_arm_positions_len)
                    else:
                        positions_msg.append(1.0)
                        positions_msg.extend(positions[label] or [])

                # Dynamics list layout per arm: [present, vx,vy,vz, ax,ay,az]
                dynamics_msg: List[float] = []
                for label in ('Left', 'Right'):
                    if dynamics[label] is None:
                        dynamics_msg.append(0.0)
                        dynamics_msg.extend([0.0] * per_arm_dynamics_len)
                    else:
                        dynamics_msg.append(1.0)
                        dynamics_msg.extend(dynamics[label] or [])

                # Send OSC
                client_positions.send_message("/arms/positions", positions_msg)
                client_dynamics.send_message("/arms/dynamics", dynamics_msg)

                # Optional small sleep to reduce CPU; adjust as needed for your Max patch responsiveness
                time.sleep(0.001)
                
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            # Release resources
            cv2.destroyAllWindows()
            cap.release()


if __name__ == "__main__":
    main()
    