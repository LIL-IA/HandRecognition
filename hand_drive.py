import time
import math
import cv2
import numpy as np
import mediapipe as mp
#TODO add hand open detection for recalibration and safety stop if one of the hands are not oppen or losed. A new recalibration will be needed.
# -------- MediaPipe setup --------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

# Landmark indices (MediaPipe Hands) -- only what's used
WRIST = 0
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def deadzone(x, dz):
    return 0.0 if abs(x) < dz else (x - np.sign(x)*dz) / (1 - dz)

def quantize(x, step):
    if step <= 0:
        return x
    return round(x / step) * step

def lerp(a, b, t):
    return a + (b - a) * t

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

def _angle_diff_deg(a, b):
    # signed shortest distance a-b in degrees
    return ((a - b + 180.0) % 360.0) - 180.0

def _hand_size_ref(lm):
    w = _v(lm, WRIST)
    idx = _v(lm, INDEX_MCP)
    mid = _v(lm, MIDDLE_MCP)
    pink = _v(lm, PINKY_MCP)
    return max((np.linalg.norm(idx-w)+np.linalg.norm(mid-w)+np.linalg.norm(pink-w))/3.0, 1e-3)

def _palm_center(lm):
    pts = [_v(lm, WRIST), _v(lm, INDEX_MCP), _v(lm, MIDDLE_MCP), _v(lm, RING_MCP), _v(lm, PINKY_MCP)]
    return np.mean(pts, axis=0)

def _handlebar_metrics(lm_left, lm_right):
    left_palm  = _palm_center(lm_left)
    right_palm = _palm_center(lm_right)
    vec_lr = right_palm - left_palm

    span_xy   = float(np.linalg.norm(vec_lr[:2]))
    span_3d   = float(np.linalg.norm(vec_lr))
    mean_z    = float((left_palm[2] + right_palm[2]) / 2.0)
    depth_diff= float(right_palm[2] - left_palm[2])  # >0: right farther than left
    size_avg  = float((_hand_size_ref(lm_left) + _hand_size_ref(lm_right)) / 2.0)
    angle_deg = math.degrees(math.atan2(vec_lr[1], vec_lr[0]))  # orientation in image plane

    return {
        'left_palm': left_palm,
        'right_palm': right_palm,
        'span_xy': span_xy,
        'span_3d': span_3d,
        'mean_z': mean_z,
        'depth_diff': depth_diff,
        'size_avg': size_avg,
        'angle_deg': angle_deg,
    }

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

        self.speed_cmd = 0
        self.direction_cmd = 0
        self.quant_step = 0.1  # -> integer steps -10..10
        self.command_hysteresis = 0.55  # fraction of a step to keep previous cmd to avoid chatter
        self.zero_lock = 0.02           # lock tiny filtered values to zero
        self.left_speed_sign = +1  # kept for compatibility with old helpers

        self.baseline = None
        self._last_full_seen = time.time()
        self._lost = False
        self.fail_timeout = 0.4   # seconds before applying safety decay
        self.decay_rate = 6.0     # how fast to decay when lost (per second)
        self.metrics_lp = None     # low-pass filter for handlebar metrics
        self.metrics_smoothing = 6.0

        # Calibration (fixed origin)
        self.calibrated = False
        self.calibrating = False
        self.calib_progress = 0.0
        self.calib_hold = 0.8           # seconds of stability to lock origin
        self.calib_stable_norm_thr = 0.08
        self.prev_metrics = None
        self.calib_requested = False

        # Ranges for bike-like control
        self.depth_range = 0.06        # z delta for full speed
        self.size_range  = 0.40        # relative size delta for full speed
        self.turn_depth_range = 0.05   # z diff between hands for full steer

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

    def reset_calibration(self):
        self.baseline = None
        self.calibrated = False
        self.calibrating = False
        self.calib_progress = 0.0
        self.prev_metrics = None
        self.metrics_lp = None

    def _stable_enough(self, metrics):
        # Normalize by typical ranges to judge stability
        if metrics is None:
            return False
        stable = False
        if self.prev_metrics is None:
            self.prev_metrics = metrics
            return False
        prev = self.prev_metrics
        keys = ['span_xy', 'mean_z', 'depth_diff', 'angle_deg', 'size_avg']
        norm = {
            'span_xy': max(prev['span_xy'], metrics['span_xy'], 1e-3),
            'mean_z':  0.05,
            'depth_diff': 0.05,
            'angle_deg': 60.0,
            'size_avg': max(prev['size_avg'], metrics['size_avg'], 1e-3),
        }
        diff = 0.0
        count = 0
        for k in keys:
            dv = abs(metrics[k] - prev[k])
            diff += dv / norm[k]
            count += 1
        diff = diff / max(count, 1)
        self.prev_metrics = metrics
        stable = diff < self.calib_stable_norm_thr
        return stable

    def filter_metrics(self, metrics, dt):
        """Low-pass filter noisy handlebar metrics to calm downstream controls."""
        if metrics is None:
            self.metrics_lp = None
            return None

        if self.metrics_lp is None:
            self.metrics_lp = metrics.copy()
            return metrics

        alpha = clamp(dt * self.metrics_smoothing, 0.0, 1.0)
        out = metrics.copy()
        for k in ('span_xy', 'mean_z', 'depth_diff', 'size_avg'):
            prev = self.metrics_lp.get(k, metrics[k])
            out[k] = float(lerp(prev, metrics[k], alpha))

        # angle needs wrap-aware interpolation to avoid jumps near +/-180 deg
        prev_ang = self.metrics_lp.get('angle_deg', metrics['angle_deg'])
        ang_delta = _angle_diff_deg(metrics['angle_deg'], prev_ang)
        out['angle_deg'] = float(prev_ang + ang_delta * alpha)

        self.metrics_lp = out
        return out

    def calibration_tick(self, metrics, dt, requested=False):
        if metrics is None:
            self.calibrating = False
            self.calib_progress = 0.0
            self.prev_metrics = None
            return {'calibrating': False, 'progress': 0.0, 'calibrated_now': False}

        # Decide if we should start calibrating
        should_start = requested or (self.baseline is None)
        if should_start and not self.calibrating:
            self.calibrating = True
            self.calib_progress = 0.0

        if not self.calibrating:
            return {'calibrating': False, 'progress': 0.0, 'calibrated_now': False}

        stable = self._stable_enough(metrics)
        if stable:
            self.calib_progress = clamp(self.calib_progress + dt / self.calib_hold, 0.0, 1.0)
        else:
            self.calib_progress = 0.0

        if self.calib_progress >= 1.0:
            self.baseline = metrics.copy()
            self.calibrated = True
            self.calibrating = False
            self.calib_progress = 0.0
            return {'calibrating': False, 'progress': 1.0, 'calibrated_now': True}

        return {'calibrating': True, 'progress': self.calib_progress, 'calibrated_now': False}

    def smooth_and_quantize(self, spd, direc, dt):
        # slightly faster response when dt grows
        t = clamp(dt * 6.0, 0.0, 1.0)
        self._f_speed = lerp(self._f_speed, spd, t)
        self._f_dir   = lerp(self._f_dir,   direc, t)

        if abs(self._f_speed) < self.zero_lock:
            self._f_speed = 0.0
        if abs(self._f_dir) < self.zero_lock:
            self._f_dir = 0.0

        target_speed = quantize(clamp(self._f_speed, -1.0, 1.0), self.quant_step)
        target_dir   = quantize(clamp(self._f_dir,   -1.0, 1.0), self.quant_step)

        stick = self.quant_step * self.command_hysteresis
        if abs(target_speed - self.speed) < stick:
            target_speed = self.speed
        if abs(target_dir - self.direction) < stick:
            target_dir = self.direction

        self.speed = target_speed
        self.direction = target_dir

        self.speed_cmd     = int(round(self.speed / self.quant_step))
        self.direction_cmd = int(round(self.direction / self.quant_step))

    def handle_visibility(self, hands_ok, dt):
        now = time.time()
        if hands_ok:
            self._last_full_seen = now
            self._lost = False
            return False

        if now - self._last_full_seen >= self.fail_timeout:
            decay = math.exp(-self.decay_rate * dt)
            self._f_speed *= decay
            self._f_dir   *= decay
            self.speed = quantize(self._f_speed, self.quant_step)
            self.direction = quantize(self._f_dir, self.quant_step)
            self.speed_cmd = int(round(self.speed / self.quant_step))
            self.direction_cmd = int(round(self.direction / self.quant_step))
            self._lost = True
            self.metrics_lp = None
            return True
        return False

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

# -------- Bike-like controls from two hands --------
def compute_bike_controls(lm_left, lm_right, state, metrics=None):
    metrics = metrics if metrics is not None else _handlebar_metrics(lm_left, lm_right)
    if state.baseline is None:
        return 0.0, 0.0, metrics, {'calibrating': True}

    base = state.baseline
    info = {'calibrating': False}

    speed_gain = 2.2
    steering_gain = 2.2

    # --- Speed: push both hands forward / enlarge handlebar span ---
    depth_push = clamp((base['mean_z'] - metrics['mean_z']) / state.depth_range, -1.0, 1.0)
    size_push  = clamp((metrics['size_avg'] - base['size_avg']) / (base['size_avg'] * state.size_range + 1e-6), -1.0, 1.0)
    span_push  = clamp((metrics['span_xy'] - base['span_xy']) / (base['span_xy'] * 0.7 + 1e-6), -1.0, 1.0)

    speed_raw = speed_gain*0.55 * depth_push + 0.35 * size_push + 0.10 * span_push
    speed_raw = clamp(deadzone(speed_raw, 0.06), -1.0, 1.0)

    # --- Steering: left hand forward / right hand back or bar tilt ---
    steer_depth = clamp(metrics['depth_diff'] / state.turn_depth_range, -1.0, 1.0)  # >0: left closer than right -> turn left
    ang_delta   = _angle_diff_deg(metrics['angle_deg'], base['angle_deg'])
    steer_angle = clamp(ang_delta / 45.0, -1.0, 1.0)

    steering_raw = steering_gain*0.70 * steer_depth + 0.30 * steer_angle
    steering_raw = clamp(deadzone(steering_raw, 0.05), -1.0, 1.0)

    info.update({
        'depth_push': depth_push,
        'size_push': size_push,
        'span_push': span_push,
        'steer_depth': steer_depth,
        'steer_angle': steer_angle,
        'baseline_angle': base['angle_deg'],
    })
    return speed_raw, steering_raw, metrics, info

# -------- Main --------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera.")
        return

    state = DriveState()
    last_print = 0.0
    prev_time = time.time()
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

            now = time.time()
            dt = now - prev_time
            prev_time = now

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            left = right = None

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

            # --- Arming (both hands open for a hold) ---
            left_open_val  = hand_openness_01(left[0].landmark)  >= 0.85 if left  else False
            right_open_val = hand_openness_01(right[0].landmark) >= 0.85 if right else False
            both_open = left_open_val and right_open_val
            p = state.update_arming(both_open)

            status_txt = f"Drive {'ENABLED' if state.enabled else 'DISABLED'}"
            txt_org = (20, 40)

            # While arming, preview the future state color
            if p is not None and both_open:
                future_enabled = not state.enabled
                txt_color = (0,255,0) if future_enabled else (0,0,255)
                bar_color = txt_color
                p_disp = p
            else:
                txt_color = (0,255,0) if state.enabled else (0,0,255)
                bar_color = txt_color
                p_disp = 1.0 if state.enabled else 0.0

            cv2.putText(frame, status_txt, txt_org, font, 0.9, txt_color, 2)
            (tw, th), _ = cv2.getTextSize(status_txt, font, 0.9, 2)
            bar_w, bar_h = 140, 18
            bar_x = txt_org[0] + tw + 12
            bar_y = txt_org[1] - th + 2
            draw_progress_bar(frame, bar_x, bar_y, bar_w, bar_h, p_disp, color=bar_color)

            # --- Bike-like control computation ---
            hands_ok = (left is not None) and (right is not None)
            lost_active = state.handle_visibility(hands_ok, dt)

            # If both hands disappear for long enough, queue a recalibration to re-lock origin
            if lost_active and left is None and right is None and not state.calib_requested:
                state.calib_requested = True

            metrics = None
            info = {}
            calib_info = {'calibrating': False, 'progress': 0.0, 'calibrated_now': False}

            if hands_ok:
                metrics_raw = _handlebar_metrics(left[0].landmark, right[0].landmark)
                metrics = state.filter_metrics(metrics_raw, dt)
                calib_info = state.calibration_tick(metrics, dt, requested=state.calib_requested)
                if calib_info.get('calibrated_now'):
                    state.calib_requested = False

                if state.baseline is not None and not state.calibrating:
                    speed_raw, dir_raw, _, info = compute_bike_controls(left[0].landmark, right[0].landmark, state, metrics=metrics)
                    state.smooth_and_quantize(speed_raw, dir_raw, dt)
                else:
                    state.smooth_and_quantize(0.0, 0.0, dt)
            else:
                state.filter_metrics(None, dt)
                state.calibration_tick(None, dt, requested=False)
                if not lost_active:
                    # short glitch: gently return to neutral
                    state.smooth_and_quantize(0.0, 0.0, dt)

            cmd_speed = state.speed_cmd if state.enabled else 0
            cmd_dir   = state.direction_cmd if state.enabled else 0

            cv2.putText(frame, f"Cmd speed: {cmd_speed:+02d} ({cmd_speed*state.quant_step:+.2f})",
                        (20, h-70), font, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Cmd steer: {cmd_dir:+02d} ({cmd_dir*state.quant_step:+.2f})",
                        (20, h-40), font, 0.7, (0, 255, 255), 2)

            if calib_info.get('calibrating'):
                ptxt = f"Calibrating origin... {int(calib_info.get('progress',0)*100):02d}%"
                cv2.putText(frame, ptxt, (20, h-100), font, 0.6, (0, 200, 0), 2)
            elif state.calib_requested and not state.calibrating:
                cv2.putText(frame, "Auto recalibration pending: show both hands and hold still",
                            (20, h-100), font, 0.6, (0, 165, 255), 2)
            elif state.baseline is None:
                cv2.putText(frame, "Hold still or press 'C' to set origin", (20, h-100), font, 0.6, (0, 165, 255), 2)
            elif lost_active:
                msg = "SAFE STOP (hands lost)"
                if state.calib_requested:
                    msg += " - auto recalibration will start when hands return"
                cv2.putText(frame, msg, (20, h-100), font, 0.6, (0, 0, 255), 2)
            elif not hands_ok:
                cv2.putText(frame, "Need both hands visible", (20, h-100), font, 0.6, (0, 165, 255), 2)

            # Print to terminal ~15 Hz
            now = time.time()
            if now - last_print >= (1/15):
                out_speed_steps = cmd_speed if state.enabled else 0
                out_dir_steps   = cmd_dir if state.enabled else 0
                print(f"drive_enabled={int(state.enabled)} speed_step={out_speed_steps:+03d} dir_step={out_dir_steps:+03d} "
                      f"speed={out_speed_steps*state.quant_step:+.3f} direction={out_dir_steps*state.quant_step:+.3f}")
                last_print = now

            cv2.imshow("Bike Hands Drive (push=throttle, left hand forward=turn left)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k in (ord('e'), ord('E')):
                state.enabled = not state.enabled
            elif k in (ord('c'), ord('C')):
                state.reset_calibration()
                state.calib_requested = True

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
