import numpy as np
from collections import deque

class FireExpertSystem:
    def __init__(self, fps=30):
        self.fps = fps
        self.buffer_size = fps * 2
        
        self.detection_history = deque(maxlen=self.buffer_size) 
        self.area_history = deque(maxlen=self.buffer_size)
        self.centroid_history = deque(maxlen=self.buffer_size)
        self.conf_history = deque(maxlen=self.buffer_size)

    def update(self, detected, box=None, conf=0.0):
        self.detection_history.append(detected)
        if detected and box is not None:
            self.area_history.append(box[2] * box[3])
            self.centroid_history.append((box[0], box[1]))
            self.conf_history.append(conf)
        else:
            self.area_history.append(0)
            self.conf_history.append(0)

    def get_fire_status(self, alarm_threshold=60): # Lowered default threshold from 80 to 60
        """
        Returns a dictionary with the current analysis.
        Adjust the internal thresholds below to be more sensitive.
        """
        score = 0
        reasons = []

        if not self.detection_history or not self.detection_history[-1]:
            return {"fire_detected": False, "score": 0, "status": "Clear", "reasons": []}

        # 1. Check Duration (Lowered confidence threshold to 0.4 for sensitivity)
        if self._check_duration(threshold=0.4, duration_sec=0.5):
            score += 40
            reasons.append("Stable detection")

        # 2. Check Growth (Lowered ratio to 1.01 - almost any growth counts)
        if self._check_growth(growth_ratio=1.01):
            score += 20
            reasons.append("Growth pattern")

        # 3. Check Flicker (Appearance frequency)
        if self._check_flicker() > 0.8:
            score += 20
            reasons.append("Flicker detected")

        # 4. Stability Veto
        if not self._check_stability(max_drift=0.2): # Relaxed drift to 0.2
            score -= 40
            reasons.append("Suspected glitch (teleporting)")

        is_alarm = score >= alarm_threshold
        return {
            "fire_detected": is_alarm,
            "score": max(0, score),
            "status": "ALARM" if is_alarm else "Sensing",
            "reasons": reasons
        }

    # Internal helper methods
    def _check_duration(self, threshold, duration_sec):
        req_frames = int(duration_sec * self.fps)
        if len(self.conf_history) < req_frames: return False
        recent = list(self.conf_history)[-req_frames:]
        return all(c >= threshold for c in recent)

    def _check_growth(self, growth_ratio):
        if len(self.area_history) < self.fps: return False
        old_area, current_area = self.area_history[0], self.area_history[-1]
        if old_area == 0: return False
        return (current_area / old_area) >= growth_ratio

    def _check_flicker(self):
        if len(self.detection_history) < self.fps: return 0.0
        h = list(self.detection_history)
        transitions = sum(1 for i in range(1, len(h)) if h[i] != h[i-1])
        return transitions / (len(h) / self.fps)

    def _check_stability(self, max_drift):
        if len(self.centroid_history) < 2: return True
        c1, c2 = self.centroid_history[-1], self.centroid_history[-2]
        return np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2) < max_drift