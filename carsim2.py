import pygame
import math
import random
import sys
import os
import threading
import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

pygame.init()

# ═══════════════════════════════════════════
#  WINDOW & DISPLAY
# ═══════════════════════════════════════════
GAME_W, GAME_H = 900, 600
PANEL_H = 150
W, H = GAME_W, GAME_H + PANEL_H
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("Car Simulator 2 — Eye Control Edition")
clock = pygame.time.Clock()

# ─── Fonts ───
font_sm = pygame.font.SysFont("consolas", 16)
font_md = pygame.font.SysFont("consolas", 22, bold=True)
font_lg = pygame.font.SysFont("consolas", 32, bold=True)
font_title = pygame.font.SysFont("consolas", 14, bold=True)

# ─── Colors ───
BG_COLOR = (22, 22, 30)
PANEL_BG = (30, 30, 42)
PANEL_BORDER = (60, 60, 80)
ROAD_COLOR = (50, 50, 60)
ROAD_LINE = (180, 180, 60)
GRASS_COLOR = (25, 50, 25)
CURB_COLOR = (200, 50, 50)

WHITE = (255, 255, 255)
GRAY = (120, 120, 120)
RED = (220, 50, 50)
GREEN = (50, 220, 80)
YELLOW = (240, 220, 60)
CYAN = (60, 220, 240)
ORANGE = (240, 140, 40)
MAGENTA = (220, 60, 200)

# Car color presets
CAR_COLORS = [
    ((220, 40, 40), "Red"),
    ((40, 180, 40), "Green"),
    ((40, 100, 220), "Blue"),
    ((220, 160, 30), "Gold"),
    ((180, 40, 180), "Purple"),
    ((40, 200, 200), "Cyan"),
    ((240, 240, 240), "White"),
    ((255, 100, 20), "Orange"),
]

# ─── Physics ───
CAR_W, CAR_H = 48, 22
MAX_SPEED = 400.0
MAX_REVERSE = -120.0
ACCEL = 500.0
BRAKE_DECEL = 800.0
REVERSE_ACCEL = 250.0
FRICTION = 150.0
TURN_SPEED = 160.0
MIN_TURN_SPEED_RATIO = 0.15
DRIFT_FACTOR = 0.92
GRASS_SLOW = 0.6
ANGRY_SPEED_CAP = 120.0  # max speed when angry


# ═══════════════════════════════════════════
#  FACE CONTROLLER – Camera + MediaPipe Tasks
# ═══════════════════════════════════════════

# MediaPipe FaceMesh landmark indices for Eye Aspect Ratio (EAR)
RIGHT_EYE = { "outer": 33, "inner": 133, "top1": 159, "top2": 158, "bot1": 145, "bot2": 153 }
LEFT_EYE = { "outer": 362, "inner": 263, "top1": 386, "top2": 385, "bot1": 374, "bot2": 380 }
RIGHT_BROW_MID = 52
LEFT_BROW_MID = 282
RIGHT_EYE_TOP = 159
LEFT_EYE_TOP = 386
RIGHT_BROW_INNER = 55
LEFT_BROW_INNER = 285

def _ear(landmarks, eye_dict):
    def _dist(i1, i2):
        a, b = landmarks[i1], landmarks[i2]
        return math.hypot(a.x - b.x, a.y - b.y)
    vert1 = _dist(eye_dict["top1"], eye_dict["bot1"])
    vert2 = _dist(eye_dict["top2"], eye_dict["bot2"])
    horiz = _dist(eye_dict["outer"], eye_dict["inner"])
    if horiz < 1e-6: return 0.3
    return (vert1 + vert2) / (2.0 * horiz)

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

class FaceController:
    """
    Threaded face-tracking controller using MediaPipe Tasks API.
    Reads camera, runs FaceLandmarker, exposes eyes & anger states.
    """
    EAR_THRESHOLD = 0.23   # slightly higher for easier winking/blinking
    ANGRY_BROW_THRESHOLD = 0.025

    def __init__(self, cam_index=0, preview_size=(240, 180)):
        self.preview_w, self.preview_h = preview_size
        self.cam_index = cam_index
        self.left_closed = False
        self.right_closed = False
        self.both_closed = False
        self.both_open = True
        self.angry = False
        self.active = False
        self.left_ear = 0.3
        self.right_ear = 0.3
        self.brow_dist = 0.05
        self._preview_frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.landmarker = None

    def start(self):
        if self._running: return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        cap = cv2.VideoCapture(self.cam_index)
        if not cap.isOpened():
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_FPS, 30)

        base_options = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        frame_timestamp_ms = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            frame_timestamp_ms += 33
            
            try:
                results = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            except Exception as e:
                time.sleep(0.01)
                continue

            if results.face_landmarks and len(results.face_landmarks) > 0:
                lm = results.face_landmarks[0]
                l_ear = _ear(lm, LEFT_EYE)
                r_ear = _ear(lm, RIGHT_EYE)

                l_closed = l_ear < self.EAR_THRESHOLD
                r_closed = r_ear < self.EAR_THRESHOLD

                brow_r = abs(lm[RIGHT_BROW_MID].y - lm[RIGHT_EYE_TOP].y)
                brow_l = abs(lm[LEFT_BROW_MID].y - lm[LEFT_EYE_TOP].y)
                avg_brow = (brow_r + brow_l) / 2.0
                inner_dist = abs(lm[RIGHT_BROW_INNER].x - lm[LEFT_BROW_INNER].x)

                is_angry = avg_brow < self.ANGRY_BROW_THRESHOLD and inner_dist < 0.06

                h_f, w_f = frame.shape[:2]
                for idx_set, color in [(LEFT_EYE, (0, 255, 0)), (RIGHT_EYE, (0, 255, 0))]:
                    for key in idx_set:
                        pt = lm[idx_set[key]]
                        cv2.circle(frame, (int(pt.x * w_f), int(pt.y * h_f)), 1, color, -1)

                for bi in [RIGHT_BROW_MID, LEFT_BROW_MID, RIGHT_BROW_INNER, LEFT_BROW_INNER]:
                    pt = lm[bi]
                    cv2.circle(frame, (int(pt.x * w_f), int(pt.y * h_f)), 2, (0, 165, 255), -1)

                with self._lock:
                    self.left_ear = l_ear
                    self.right_ear = r_ear
                    self.left_closed = l_closed
                    self.right_closed = r_closed
                    self.both_closed = l_closed and r_closed
                    self.both_open = not l_closed and not r_closed
                    self.angry = is_angry
                    self.brow_dist = avg_brow
                    self.active = True

                b_status = "CLOSED" if (l_closed and r_closed) else ("OPEN" if (not l_closed and not r_closed) else "WINK")
                status = "ANGRY" if is_angry else b_status
                
                if is_angry:
                    cv2.putText(frame, "ANGRY!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif self.both_closed:
                    cv2.putText(frame, "BRAKING (BOTH CLOSED)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                elif self.left_closed:
                    cv2.putText(frame, "STEER LEFT (WINK)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                elif self.right_closed:
                    cv2.putText(frame, "STEER RIGHT (WINK)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                else:
                    cv2.putText(frame, "ACCELERATING (BOTH OPEN)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                with self._lock:
                    self.active = False
                    self.both_open = True
                    self.both_closed = False
                    self.left_closed = False
                    self.right_closed = False

            small = cv2.resize(frame, (self.preview_w, self.preview_h))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            with self._lock:
                self._preview_frame = small_rgb

            time.sleep(0.01)

        cap.release()
        try:
            self.landmarker.close()
        except:
            pass

    def get_preview_surface(self):
        with self._lock:
            if self._preview_frame is None:
                return None
            frame = self._preview_frame.copy()
        return pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    def get_state(self):
        with self._lock:
            return {
                "active": self.active,
                "left_closed": self.left_closed,
                "right_closed": self.right_closed,
                "both_closed": self.both_closed,
                "both_open": self.both_open,
                "angry": self.angry,
                "left_ear": self.left_ear,
                "right_ear": self.right_ear,
                "brow_dist": self.brow_dist,
            }


# ═══════════════════════════════════════════
#  TRACK
# ═══════════════════════════════════════════
class Track:
    def __init__(self):
        self.cx, self.cy = GAME_W // 2, GAME_H // 2
        self.rx_outer, self.ry_outer = 380, 240
        self.rx_inner, self.ry_inner = 220, 110
        self.surface = pygame.Surface((GAME_W, GAME_H))
        self._render()

    def _render(self):
        self.surface.fill(GRASS_COLOR)
        # outer curb
        pygame.draw.ellipse(self.surface, CURB_COLOR,
            (self.cx - self.rx_outer - 6, self.cy - self.ry_outer - 6,
             (self.rx_outer + 6) * 2, (self.ry_outer + 6) * 2))
        # road
        pygame.draw.ellipse(self.surface, ROAD_COLOR,
            (self.cx - self.rx_outer, self.cy - self.ry_outer,
             self.rx_outer * 2, self.ry_outer * 2))
        # inner curb
        pygame.draw.ellipse(self.surface, CURB_COLOR,
            (self.cx - self.rx_inner + 6, self.cy - self.ry_inner + 6,
             (self.rx_inner - 6) * 2, (self.ry_inner - 6) * 2))
        # grass center
        pygame.draw.ellipse(self.surface, GRASS_COLOR,
            (self.cx - self.rx_inner, self.cy - self.ry_inner,
             self.rx_inner * 2, self.ry_inner * 2))
        # dashed center line
        mid_rx = (self.rx_outer + self.rx_inner) / 2
        mid_ry = (self.ry_outer + self.ry_inner) / 2
        for i in range(60):
            if i % 3 == 0:
                continue
            a1 = math.radians(i * 6)
            a2 = math.radians((i + 1) * 6)
            x1 = self.cx + math.cos(a1) * mid_rx
            y1 = self.cy + math.sin(a1) * mid_ry
            x2 = self.cx + math.cos(a2) * mid_rx
            y2 = self.cy + math.sin(a2) * mid_ry
            pygame.draw.line(self.surface, ROAD_LINE, (x1, y1), (x2, y2), 2)
        # start/finish line
        pygame.draw.line(self.surface, WHITE,
                         (self.cx + self.rx_inner, self.cy),
                         (self.cx + self.rx_outer, self.cy), 4)

    def is_on_road(self, x, y):
        dx = (x - self.cx)
        dy = (y - self.cy)
        d_outer = (dx / self.rx_outer) ** 2 + (dy / self.ry_outer) ** 2
        d_inner = (dx / self.rx_inner) ** 2 + (dy / self.ry_inner) ** 2
        return d_inner >= 1.0 and d_outer <= 1.0

    def draw(self, surf):
        surf.blit(self.surface, (0, 0))


# ═══════════════════════════════════════════
#  PARTICLE
# ═══════════════════════════════════════════
class Particle:
    def __init__(self, x, y, kind="smoke"):
        self.x = x
        self.y = y
        self.kind = kind
        self.life = 1.0
        if kind == "smoke":
            self.decay = random.uniform(1.5, 3.0)
            self.size = random.randint(3, 6)
            self.vx = random.uniform(-15, 15)
            self.vy = random.uniform(-15, 15)
        elif kind == "tire":
            self.decay = 0.3
            self.size = 2
            self.vx = self.vy = 0
        elif kind == "spark":
            self.decay = random.uniform(3.0, 6.0)
            self.size = random.randint(1, 3)
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(40, 120)
            self.vx = math.cos(ang) * spd
            self.vy = math.sin(ang) * spd

    def update(self, dt):
        self.life -= self.decay * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.kind == "smoke":
            self.vx *= 0.95
            self.vy *= 0.95

    def draw(self, surf):
        if self.life <= 0:
            return
        alpha = max(0, min(255, int(self.life * 180)))
        if self.kind == "smoke":
            c = (alpha // 2, alpha // 2, alpha // 2)
        elif self.kind == "tire":
            c = (30, 30, 30)
        elif self.kind == "spark":
            c = (255, min(255, 100 + int(self.life * 155)), 30)
        pygame.draw.circle(surf, c, (int(self.x), int(self.y)), self.size)


# ═══════════════════════════════════════════
#  CAR
# ═══════════════════════════════════════════
class Car:
    def __init__(self, color_idx=0):
        self.color_idx = color_idx
        self.body_color = CAR_COLORS[color_idx][0]
        self.reset()

    def reset(self):
        self.x = float(GAME_W // 2 + 300)
        self.y = float(GAME_H // 2 - 10)
        self.angle = 90.0
        self.speed = 0.0
        self.drift_angle = 0.0
        self.on_road = True
        self.lap_timer = 0.0
        self.best_lap = None
        self.laps = 0
        self.crossed_half = False
        self.last_checkpoint = False
        self.headlights_on = True
        self.angry_mode = False  # speed-limited when angry

    def set_color(self, idx):
        self.color_idx = idx % len(CAR_COLORS)
        self.body_color = CAR_COLORS[self.color_idx][0]

    def _make_surface(self):
        s = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
        pygame.draw.rect(s, self.body_color, (4, 2, CAR_W - 8, CAR_H - 4), border_radius=5)
        darker = tuple(max(0, c - 80) for c in self.body_color)
        pygame.draw.rect(s, darker, (CAR_W - 14, 4, 8, CAR_H - 8), border_radius=3)
        if self.headlights_on:
            pygame.draw.circle(s, (255, 255, 180), (CAR_W - 4, 5), 3)
            pygame.draw.circle(s, (255, 255, 180), (CAR_W - 4, CAR_H - 5), 3)
        pygame.draw.circle(s, (200, 0, 0), (4, 4), 3)
        pygame.draw.circle(s, (200, 0, 0), (4, CAR_H - 4), 3)
        return s

    def update(self, dt, track, particles, face_input=None):
        """
        face_input: dict with keys  accelerate, brake, steer_left, steer_right, angry
                    or None to fall back to keyboard.
        """
        keys = pygame.key.get_pressed()

        # ── Determine inputs (keyboard OR face) ──
        inp_accel = False
        inp_brake = False
        inp_left = False
        inp_right = False
        is_angry = False

        if face_input:
            inp_accel = face_input.get("accelerate", False)
            inp_brake = face_input.get("brake", False)
            inp_left = face_input.get("steer_left", False)
            inp_right = face_input.get("steer_right", False)
            is_angry = face_input.get("angry", False)
        else:
            inp_accel = keys[pygame.K_w] or keys[pygame.K_UP]
            inp_brake = keys[pygame.K_s] or keys[pygame.K_DOWN]
            inp_left = keys[pygame.K_a] or keys[pygame.K_LEFT]
            inp_right = keys[pygame.K_d] or keys[pygame.K_RIGHT]

        # also allow keyboard override even in eye mode
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            inp_accel = True
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            inp_brake = True
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            inp_left = True
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            inp_right = True

        self.angry_mode = is_angry

        # ── acceleration / braking ──
        accel_input = 0
        braking = False
        if inp_accel:
            if self.speed < 0:
                accel_input = BRAKE_DECEL
                braking = True
            else:
                accel_input = ACCEL
        elif inp_brake:
            if self.speed > 5:
                accel_input = -BRAKE_DECEL
                braking = True
            else:
                accel_input = -REVERSE_ACCEL

        self.speed += accel_input * dt

        # friction
        if accel_input == 0:
            if abs(self.speed) < FRICTION * dt:
                self.speed = 0
            elif self.speed > 0:
                self.speed -= FRICTION * dt
            else:
                self.speed += FRICTION * dt

        # grass penalty
        self.on_road = track.is_on_road(self.x, self.y)
        if not self.on_road:
            self.speed *= (1.0 - GRASS_SLOW * dt)

        # angry penalty – cap the max speed
        effective_max = ANGRY_SPEED_CAP if self.angry_mode else MAX_SPEED
        self.speed = max(MAX_REVERSE, min(self.speed, effective_max))

        # extra deceleration when angry and above cap
        if self.angry_mode and self.speed > ANGRY_SPEED_CAP:
            self.speed -= 300 * dt
            self.speed = max(self.speed, ANGRY_SPEED_CAP)

        # ── steering ──
        turn = 0
        speed_ratio = min(abs(self.speed) / MAX_SPEED, 1.0)
        effective_turn = TURN_SPEED * max(MIN_TURN_SPEED_RATIO, speed_ratio)
        if inp_left:
            turn = effective_turn
        if inp_right:
            turn = -effective_turn
        if self.speed < 0:
            turn = -turn
        self.angle += turn * dt

        # drift
        is_drifting = abs(turn) > 0 and abs(self.speed) > MAX_SPEED * 0.55
        if is_drifting:
            self.drift_angle = turn * 0.3
        else:
            self.drift_angle *= DRIFT_FACTOR

        # movement
        rad = math.radians(self.angle + self.drift_angle * 0.2)
        self.x += math.cos(rad) * self.speed * dt
        self.y -= math.sin(rad) * self.speed * dt
        self.x = max(10, min(self.x, GAME_W - 10))
        self.y = max(10, min(self.y, GAME_H - 10))

        # particles
        if braking and abs(self.speed) > 30:
            for offset in [-8, 8]:
                ox = self.x - math.cos(rad) * 18 + math.sin(rad) * offset
                oy = self.y + math.sin(rad) * 18 + math.cos(rad) * offset
                particles.append(Particle(ox, oy, "tire"))
        if is_drifting:
            for _ in range(2):
                ox = self.x - math.cos(rad) * 20 + random.uniform(-5, 5)
                oy = self.y + math.sin(rad) * 20 + random.uniform(-5, 5)
                particles.append(Particle(ox, oy, "smoke"))
        if abs(self.speed) > MAX_SPEED * 0.85:
            if random.random() < 0.3:
                ex = self.x - math.cos(rad) * 24
                ey = self.y + math.sin(rad) * 24
                particles.append(Particle(ex, ey, "spark"))

        # lap detection
        self.lap_timer += dt
        dx = self.x - track.cx
        dy = self.y - track.cy
        if dy > 0 and abs(dx) < 60:
            self.crossed_half = True
        finish_x = track.cx + (track.rx_outer + track.rx_inner) / 2
        if abs(self.x - finish_x) < 30 and abs(self.y - track.cy) < 40:
            if not self.last_checkpoint and self.crossed_half:
                self.laps += 1
                if self.best_lap is None or self.lap_timer < self.best_lap:
                    self.best_lap = self.lap_timer
                self.lap_timer = 0.0
                self.crossed_half = False
            self.last_checkpoint = True
        else:
            self.last_checkpoint = False

    def draw(self, surf):
        car_surf = self._make_surface()
        rotated = pygame.transform.rotate(car_surf, self.angle + self.drift_angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        surf.blit(rotated, rect)
        # headlight glow
        if self.headlights_on and abs(self.speed) > 5:
            rad = math.radians(self.angle)
            glow_surf = pygame.Surface((60, 60), pygame.SRCALPHA)
            alpha = min(80, int(abs(self.speed) / MAX_SPEED * 80))
            pygame.draw.circle(glow_surf, (255, 255, 180, alpha), (30, 30), 30)
            gx = self.x + math.cos(rad) * 30 - 30
            gy = self.y - math.sin(rad) * 30 - 30
            surf.blit(glow_surf, (gx, gy), special_flags=pygame.BLEND_ADD)


# ═══════════════════════════════════════════
#  DASHBOARD DRAWING
# ═══════════════════════════════════════════
def draw_speedometer(surf, x, y, speed, max_spd):
    radius = 52
    pygame.draw.arc(surf, (50, 50, 60),
                    (x - radius, y - radius, radius * 2, radius * 2),
                    math.radians(0), math.radians(180), 4)
    for i in range(11):
        a = math.radians(180 - i * 18)
        x1 = x + math.cos(a) * (radius - 8)
        y1 = y - math.sin(a) * (radius - 8)
        x2 = x + math.cos(a) * radius
        y2 = y - math.sin(a) * radius
        col = GREEN if i < 7 else (YELLOW if i < 9 else RED)
        pygame.draw.line(surf, col, (x1, y1), (x2, y2), 2)
    ratio = min(abs(speed) / max_spd, 1.0)
    needle_a = math.radians(180 - ratio * 180)
    nx = x + math.cos(needle_a) * (radius - 14)
    ny = y - math.sin(needle_a) * (radius - 14)
    pygame.draw.line(surf, RED, (x, y), (nx, ny), 3)
    pygame.draw.circle(surf, WHITE, (x, y), 5)
    spd_text = font_md.render(f"{int(abs(speed))} km/h", True, WHITE)
    surf.blit(spd_text, (x - spd_text.get_width() // 2, y + 8))


def draw_gear_indicator(surf, x, y, speed):
    if abs(speed) < 2:
        gear, color = "N", GRAY
    elif speed < 0:
        gear, color = "R", RED
    elif speed < 80:
        gear, color = "1", GREEN
    elif speed < 160:
        gear, color = "2", GREEN
    elif speed < 260:
        gear, color = "3", YELLOW
    elif speed < 340:
        gear, color = "4", YELLOW
    else:
        gear, color = "5", RED
    label = font_sm.render("GEAR", True, GRAY)
    surf.blit(label, (x - label.get_width() // 2, y - 20))
    g_text = font_lg.render(gear, True, color)
    surf.blit(g_text, (x - g_text.get_width() // 2, y))


def draw_minimap(surf, x, y, track, car):
    mm_w, mm_h = 100, 60
    pygame.draw.rect(surf, (20, 20, 30), (x, y, mm_w, mm_h), border_radius=6)
    pygame.draw.rect(surf, PANEL_BORDER, (x, y, mm_w, mm_h), 1, border_radius=6)
    cx = x + mm_w // 2
    cy = y + mm_h // 2
    sx = mm_w / GAME_W * 0.85
    sy = mm_h / GAME_H * 0.85
    orx = int(track.rx_outer * sx)
    ory = int(track.ry_outer * sy)
    irx = int(track.rx_inner * sx)
    iry = int(track.ry_inner * sy)
    pygame.draw.ellipse(surf, (60, 60, 70), (cx - orx, cy - ory, orx * 2, ory * 2), 1)
    pygame.draw.ellipse(surf, (60, 60, 70), (cx - irx, cy - iry, irx * 2, iry * 2), 1)
    car_mx = cx + int((car.x - track.cx) * sx)
    car_my = cy + int((car.y - track.cy) * sy)
    pygame.draw.circle(surf, car.body_color, (car_mx, car_my), 3)


def draw_eye_status(surf, x, y, face_state, eye_mode):
    """Draw the eye-control status indicator on the panel."""
    # Mode badge
    if eye_mode:
        mode_text = "👁  EYE CONTROL"
        mode_color = CYAN
    else:
        mode_text = "⌨  KEYBOARD"
        mode_color = GRAY
    badge = font_md.render(mode_text, True, mode_color)
    surf.blit(badge, (x, y))

    if not eye_mode:
        hint = font_sm.render("Press E to enable eye control", True, (80, 80, 100))
        surf.blit(hint, (x, y + 24))
        return

    if not face_state["active"]:
        no_face = font_sm.render("No face detected...", True, RED)
        surf.blit(no_face, (x, y + 25))
        return

    # EAR values
    ly = y + 26
    l_col = RED if face_state["left_closed"] else GREEN
    r_col = RED if face_state["right_closed"] else GREEN

    # Left eye indicator
    l_label = "L-EYE: CLOSED" if face_state["left_closed"] else "L-EYE: OPEN"
    surf.blit(font_sm.render(l_label, True, l_col), (x, ly))
    # EAR bar
    ear_w = int(face_state["left_ear"] * 200)
    pygame.draw.rect(surf, (40, 40, 50), (x + 130, ly + 2, 60, 10))
    pygame.draw.rect(surf, l_col, (x + 130, ly + 2, min(ear_w, 60), 10))

    ly += 18
    # Right eye indicator
    r_label = "R-EYE: CLOSED" if face_state["right_closed"] else "R-EYE: OPEN"
    surf.blit(font_sm.render(r_label, True, r_col), (x, ly))
    ear_w = int(face_state["right_ear"] * 200)
    pygame.draw.rect(surf, (40, 40, 50), (x + 130, ly + 2, 60, 10))
    pygame.draw.rect(surf, r_col, (x + 130, ly + 2, min(ear_w, 60), 10))

    ly += 18
    # Action summary
    if face_state["both_closed"]:
        action = "⏹ BRAKING"
        ac = RED
    elif face_state["left_closed"]:
        action = "◀ STEER LEFT"
        ac = CYAN
    elif face_state["right_closed"]:
        action = "▶ STEER RIGHT"
        ac = CYAN
    elif face_state["both_open"]:
        action = "▲ ACCELERATING"
        ac = GREEN
    else:
        action = "—"
        ac = GRAY
    surf.blit(font_sm.render(action, True, ac), (x, ly))

    # Angry indicator
    ly += 18
    if face_state["angry"]:
        angry_text = font_sm.render("😠 ANGRY → SLOW MODE", True, ORANGE)
        surf.blit(angry_text, (x, ly))
    else:
        calm_text = font_sm.render("😊 Calm", True, (80, 140, 80))
        surf.blit(calm_text, (x, ly))


def draw_panel(surf, car, track, face_state, eye_mode):
    panel_y = GAME_H
    pygame.draw.rect(surf, PANEL_BG, (0, panel_y, W, PANEL_H))
    pygame.draw.line(surf, PANEL_BORDER, (0, panel_y), (W, panel_y), 2)

    # speedometer
    draw_speedometer(surf, 70, panel_y + 80, car.speed, MAX_SPEED)

    # gear
    draw_gear_indicator(surf, 170, panel_y + 62, car.speed)

    # lap info
    ix = 230
    surf.blit(font_md.render(f"LAP {car.laps}", True, CYAN), (ix, panel_y + 12))
    surf.blit(font_sm.render(f"Time: {car.lap_timer:.1f}s", True, WHITE), (ix, panel_y + 40))
    best_s = f"{car.best_lap:.1f}s" if car.best_lap else "--.-s"
    surf.blit(font_sm.render(f"Best: {best_s}", True, YELLOW), (ix, panel_y + 60))
    road_s = "ON ROAD" if car.on_road else "OFF ROAD"
    surf.blit(font_sm.render(road_s, True, GREEN if car.on_road else RED), (ix, panel_y + 84))

    if car.angry_mode:
        surf.blit(font_sm.render("⚠ SLOW MODE", True, ORANGE), (ix, panel_y + 104))

    # minimap
    draw_minimap(surf, 340, panel_y + 14, track, car)

    # eye control status
    draw_eye_status(surf, 470, panel_y + 8, face_state, eye_mode)

    # controls (compact)
    cx = 700
    ctrls = [
        "W/↑ Accel   A/← Left",
        "S/↓ Brake   D/→ Right",
        "E  Eye Mode  C Color",
        "H  Lights   R Reset",
    ]
    surf.blit(font_title.render("CONTROLS", True, GRAY), (cx, panel_y + 8))
    for i, line in enumerate(ctrls):
        surf.blit(font_sm.render(line, True, (100, 100, 120)), (cx, panel_y + 24 + i * 16))

    # car color
    cname = CAR_COLORS[car.color_idx][1]
    surf.blit(font_sm.render(f"Car: {cname}", True, car.body_color), (cx, panel_y + 100))


# ═══════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════
def main():
    track = Track()
    car = Car(0)
    particles = []
    face_ctrl = FaceController(cam_index=0, preview_size=(160, 120))
    eye_mode = False
    face_state = {
        "active": False, "left_closed": False, "right_closed": False,
        "both_closed": False, "both_open": True, "angry": False,
        "left_ear": 0.3, "right_ear": 0.3, "brow_dist": 0.05,
    }

    running = True
    preview_surf = None
    preview_timer = 0

    while running:
        dt = clock.tick(60) / 1000.0
        dt = min(dt, 0.05)

        # ── Events ──
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    car.reset()
                    particles.clear()
                elif e.key == pygame.K_c:
                    car.set_color(car.color_idx + 1)
                elif e.key == pygame.K_h:
                    car.headlights_on = not car.headlights_on
                elif e.key == pygame.K_e:
                    eye_mode = not eye_mode
                    if eye_mode:
                        face_ctrl.start()
                        print("[INFO] Eye control ENABLED – look at the camera!")
                    else:
                        face_ctrl.stop()
                        print("[INFO] Eye control DISABLED – keyboard mode")

        # ── Face state ──
        face_input = None
        if eye_mode:
            face_state = face_ctrl.get_state()
            if face_state["active"]:
                face_input = {
                    "accelerate": face_state["both_open"] and not face_state["angry"],
                    "brake": face_state["both_closed"],
                    "steer_left": face_state["left_closed"] and not face_state["right_closed"],
                    "steer_right": face_state["right_closed"] and not face_state["left_closed"],
                    "angry": face_state["angry"],
                }
                # When angry but eyes open, still accelerate (just capped)
                if face_state["angry"] and face_state["both_open"]:
                    face_input["accelerate"] = True

            # update preview every ~100ms
            preview_timer += dt
            if preview_timer > 0.1:
                preview_timer = 0
                preview_surf = face_ctrl.get_preview_surface()

        # ── Update ──
        car.update(dt, track, particles, face_input)

        for p in particles:
            p.update(dt)
        particles = [p for p in particles if p.life > 0]
        if len(particles) > 500:
            particles = particles[-500:]

        # ── Draw ──
        screen.fill(BG_COLOR)
        track.draw(screen)

        for p in particles:
            if p.kind == "tire":
                p.draw(screen)
        car.draw(screen)
        for p in particles:
            if p.kind != "tire":
                p.draw(screen)

        # camera preview (top-left corner, with border)
        if eye_mode and preview_surf:
            pw, ph = preview_surf.get_size()
            preview_x, preview_y = 8, 8
            # dark bg + border
            pygame.draw.rect(screen, (10, 10, 15), (preview_x - 2, preview_y - 2, pw + 4, ph + 4),
                             border_radius=4)
            pygame.draw.rect(screen, CYAN if face_state["active"] else RED,
                             (preview_x - 2, preview_y - 2, pw + 4, ph + 4), 2, border_radius=4)
            screen.blit(preview_surf, (preview_x, preview_y))
            # label
            cam_label = font_title.render("CAMERA", True, CYAN)
            screen.blit(cam_label, (preview_x + 4, preview_y + ph + 4))

        # angry overlay flash
        if car.angry_mode:
            overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
            pulse = int(abs(math.sin(pygame.time.get_ticks() / 300)) * 30)
            overlay.fill((200, 50, 0, pulse))
            screen.blit(overlay, (0, 0))
            angry_banner = font_lg.render("😠 ANGRY — SLOW MODE", True, ORANGE)
            screen.blit(angry_banner, (GAME_W // 2 - angry_banner.get_width() // 2, 12))

        # panel
        draw_panel(screen, car, track, face_state, eye_mode)

        # title
        title_surf = font_sm.render("CAR SIMULATOR 2 — EYE CONTROL", True, (80, 80, 100))
        screen.blit(title_surf, (GAME_W - title_surf.get_width() - 10, 8))

        pygame.display.flip()

    # cleanup
    face_ctrl.stop()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()