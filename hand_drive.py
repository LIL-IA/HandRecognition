import time
import math
import cv2
import numpy as np
import mediapipe as mp

# -------- MediaPipe setup --------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# Landmark indices (MediaPipe Hands) — only what's used
WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

def ema(prev, new, alpha=0.25):
    return alpha*new + (1-alpha)*prev

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def deadzone(x, dz):
    return 0.0 if abs(x) < dz else (x - np.sign(x)*dz) / (1 - dz)

def draw_progress_bar(img, x, y, w, h, p, color=(0,255,255)):
    p = clamp(p, 0.0, 1.0)
    cv2.rectangle(img, (x,y), (x+w, y+h), (50,50,50), 1)
    cv2.rectangle(img, (x,y), (x+int(w*p), y+h), color, -1)

def _v(lm, i):
    p = lm[i]
    return np.array([p.x, p.y, p.z], dtype=np.float32)

def _norm(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-8)

def _angle_deg(u, v):
    u = _norm(u); v = _norm(v)
    d = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return math.degrees(math.acos(d))

def _hand_size_ref(lm):
    w = _v(lm, WRIST)
    idx = _v(lm, INDEX_MCP)
    mid = _v(lm, MIDDLE_MCP)
    pink = _v(lm, PINKY_MCP)
    return max((np.linalg.norm(idx-w)+np.linalg.norm(mid-w)+np.linalg.norm(pink-w))/3.0, 1e-3)

def _palm_center(lm):
    pts = [_v(lm, WRIST), _v(lm, INDEX_MCP), _v(lm, MIDDLE_MCP), _v(lm, RING_MCP), _v(lm, PINKY_MCP)]
    return np.mean(pts, axis=0)

# -------- Finger features needed for openness and simple fist gate --------
INDEX  = (INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP)
MIDDLE = (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
RING   = (RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP)
PINKY  = (PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP)

def _curl_pip_deg(lm, joint_ids):
    MCP, PIP, DIP, TIP = joint_ids
    v1 = _v(lm, PIP) - _v(lm, MCP)
    v2 = _v(lm, TIP) - _v(lm, PIP)
    return _angle_deg(v1, v2)

def _tip_palm_dist(lm, tip_idx):
    c = _palm_center(lm)
    return np.linalg.norm(_v(lm, tip_idx) - c)

def _finger_features(lm, joint_ids):
    ref = _hand_size_ref(lm)
    MCP, PIP, DIP, TIP = joint_ids
    return {'tip_palm_n': _tip_palm_dist(lm, TIP)/ref}

def _all_finger_feats(lm):
    return {
        'index':  _finger_features(lm, INDEX),
        'middle': _finger_features(lm, MIDDLE),
        'ring':   _finger_features(lm, RING),
        'pinky':  _finger_features(lm, PINKY),
    }

def _map01(x, a0, a1):
    return float(np.clip((x - a0) / (a1 - a0), 0.0, 1.0))

# -------- Openness (speed magnitude) --------
def finger_openness_01(f):
    # Only distance-to-palm, tuned range
    return _map01(f['tip_palm_n'], 0.50, 0.90)

def hand_openness_01(lm):
    feats = _all_finger_feats(lm)
    vals = [finger_openness_01(f) for f in feats.values()]
    return float(np.mean(vals)) if vals else 0.0

# -------- Facing camera vs back-of-hand (speed sign) --------
def palm_facing_sign(lm):
    tips  = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    mcps  = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    tip_z = float(np.mean([lm[i].z for i in tips]))
    mcp_z = float(np.mean([lm[i].z for i in mcps]))
    dz = mcp_z - tip_z   # >0: tips closer than knuckles -> palm to camera
    THR = 0.015
    if dz > THR:  return +1, dz
    if dz < -THR: return -1, dz
    return 0, dz

# -------- Right-hand steering --------
def direction_from_right_fist_roll(lm):
    a = lm[INDEX_MCP]; b = lm[PINKY_MCP]
    dx = (b.x - a.x); dy = (b.y - a.y)
    ang_deg = math.degrees(math.atan2(dy, dx))
    ang_deg = ((ang_deg - 90.0 + 180.0) % 360.0) - 180.0
    ang_deg_clamped = clamp(ang_deg, -50.0, 50.0)
    val = deadzone(ang_deg_clamped / 50.0, 4.0/50.0)
    return clamp(val, -1.0, 1.0), ang_deg

def right_is_fist_simple(lm):
    curls = [_curl_pip_deg(lm, fids) for fids in (INDEX, MIDDLE, RING, PINKY)]
    return sum(1 for c in curls if c >= 45.0) >= 3

# -------- State --------
class DriveState:
    def __init__(self):
        self.enabled = False
        self._arming = False
        self._t0     = None
        self._hold   = 1.5

        self.speed = 0.0
        self.direction = 0.0
        self._f_speed = 0.0
        self._f_dir   = 0.0

        self.left_speed_sign = +1  # hold last sign when facing ambiguous

    def update_arming(self, both_open):
        if both_open:
            if not self._arming:
                self._arming = True
                self._t0 = time.time()
            elapsed = time.time() - self._t0
            if elapsed >= self._hold:
                self.enabled = not self.enabled
                self._arming = False
                self._t0 = None
                return None
            return clamp(elapsed / self._hold, 0.0, 1.0)
        else:
            self._arming = False
            self._t0 = None
            return None

    def set_speed_dir(self, spd, direc):
        self._f_speed = ema(self._f_speed, spd, 0.25)
        self._f_dir   = ema(self._f_dir,   direc, 0.25)
        self.speed = clamp(self._f_speed, -1, 1)
        self.direction = clamp(self._f_dir, -1, 1)

# -------- Speed from left openness + facing --------
def speed_from_left_openness_facing(lm, state):
    openv = hand_openness_01(lm)
    mag = clamp(1.0 - openv, 0.0, 1.0)
    sign, dz = palm_facing_sign(lm)
    if sign != 0:
        state.left_speed_sign = sign
    s = state.left_speed_sign * mag
    s = 0.0 if abs(s) < 0.03 else s
    return s, openv, dz, state.left_speed_sign

# -------- Main --------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera.")
        return

    state = DriveState()
    last_print = 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        model_complexity=1,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            left = right = None
            left_open = right_open = False

            if res.multi_hand_landmarks and res.multi_handedness:
                packs = []
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = hd.classification[0].label
                    packs.append((lm, label))
                for lm, label in packs:
                    if label == 'Left' and left is None:
                        left = (lm, label)
                    elif label == 'Right' and right is None:
                        right = (lm, label)

                for pack in (left, right):
                    if pack:
                        mp_draw.draw_landmarks(frame, pack[0], mp_hands.HAND_CONNECTIONS)

                # --- UI header: Drive status text + progress bar right next to it ---
                status_txt = f"Drive {'ENABLED' if state.enabled else 'DISABLED'}"
                txt_org = (20, 40)

                # compute openness to decide if we are arming (both very open)
                left_open_val  = hand_openness_01(left[0].landmark)  >= 0.85 if left  else False
                right_open_val = hand_openness_01(right[0].landmark) >= 0.85 if right else False
                both_open = left_open_val and right_open_val

                # get arming progress (None when not arming or after toggle happened)
                p = state.update_arming(both_open)

                # while arming, show the *future* state color; otherwise show current state color
                if p is not None and both_open:
                    future_enabled = not state.enabled
                    txt_color = (0,255,0) if future_enabled else (0,0,255)   # green/red
                    bar_color = txt_color
                    p_disp = p                                               # show hold progress
                else:
                    future_enabled = state.enabled
                    txt_color = (0,255,0) if state.enabled else (0,0,255)    # green/red
                    bar_color = txt_color
                    p_disp = 1.0 if state.enabled else 0.0                   # full if enabled, empty if disabled

                # draw the label in its color
                cv2.putText(frame, status_txt, txt_org, font, 0.9, txt_color, 2)

                # place the bar just to the right of the text
                (tw, th), _ = cv2.getTextSize(status_txt, font, 0.9, 2)
                bar_w, bar_h = 140, 18
                bar_x = txt_org[0] + tw + 12
                bar_y = txt_org[1] - th + 2
                draw_progress_bar(frame, bar_x, bar_y, bar_w, bar_h, p_disp, color=bar_color)


                # --- LEFT: speed from openness + facing (palm vs back) ---
                spd_raw = 0.0
                speed_y = h - 50
                if left:
                    lmL = left[0].landmark
                    spd_raw, _, _, _ = speed_from_left_openness_facing(lmL, state)
                    cv2.putText(frame, f"Speed = {spd_raw:+.2f}",
                                (20, speed_y), font, 0.7, (255, 0, 0), 2)  # blue
                else:
                    cv2.putText(frame, "Speed = N/A (no left hand)",
                                (20, speed_y), font, 0.7, (255, 0, 0), 2)  # blue

                # --- RIGHT: direction from fist roll (gate with simple fist) ---
                dir_raw = 0.0
                dir_y = h - 20
                if right:
                    lmR = right[0].landmark
                    if right_is_fist_simple(lmR):
                        dir_raw, _ = direction_from_right_fist_roll(lmR)
                        cv2.putText(frame, f"Direction = {dir_raw:+.2f}",
                                    (20, dir_y), font, 0.7, (0, 255, 255), 2)  # yellow
                    else:
                        cv2.putText(frame, "Direction = 0.00 (right not fist)",
                                    (20, dir_y), font, 0.7, (0, 255, 255), 2)  # yellow
                else:
                    cv2.putText(frame, "Direction = N/A (no right hand)",
                                (20, dir_y), font, 0.7, (0, 255, 255), 2)      # yellow

                state.set_speed_dir(spd_raw, dir_raw)

            else:
                # No hands at all: still show status bar + placeholders
                status_txt = f"Drive {'ENABLED' if state.enabled else 'DISABLED'}"
                txt_org = (20, 40)
                cv2.putText(frame, status_txt, txt_org, font, 0.9, (0,255,255), 2)
                (tw, th), baseline = cv2.getTextSize(status_txt, font, 0.9, 2)
                bar_w, bar_h = 140, 18
                bar_x = txt_org[0] + tw + 12
                bar_y = txt_org[1] - th + 2
                # not arming (no hands), show full if enabled else empty
                p_disp = 1.0 if state.enabled else 0.0
                bar_color = (0,255,0) if state.enabled else (0,0,255)
                draw_progress_bar(frame, bar_x, bar_y, bar_w, bar_h, p_disp, color=bar_color)

                # Placeholders
                cv2.putText(frame, "Speed = N/A (no left hand)",
                            (20, h-50), font, 0.7, (255, 0, 0), 2)          # blue
                cv2.putText(frame, "Direction = N/A (no right hand)",
                            (20, h-20), font, 0.7, (0, 255, 255), 2)        # yellow

            # Print to terminal ~15 Hz
            now = time.time()
            if now - last_print >= (1/15):
                out_speed = state.speed if state.enabled else 0.0
                out_dir   = state.direction if state.enabled else 0.0
                print(f"drive_enabled={int(state.enabled)} speed={out_speed:+.3f} direction={out_dir:+.3f}")
                last_print = now

            cv2.imshow("Hand Drive (speed=open, PALM→cam=forward, BACK→cam=reverse)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k in (ord('e'), ord('E')):
                state.enabled = not state.enabled

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
