import numpy as np
import time
from collections import deque

class FireExpertSystem:
    def __init__(self, fps=30, alarm_threshold=80):
        self.fps = fps
        self.alarm_threshold = alarm_threshold
        self.buffer_size = fps * 4 # 2 seconds of memory
        
        # Buffers for history
        self.detection_history = deque(maxlen=self.buffer_size) 
        self.area_history = deque(maxlen=self.buffer_size)
        self.centroid_history = deque(maxlen=self.buffer_size)
        self.conf_history = deque(maxlen=self.buffer_size)

    def update(self, detected, box=None, conf=0.0):
        """Updates internal state with data from the latest frame."""
        self.detection_history.append(detected)
        if detected and box is not None:
            # box format: [x_center, y_center, width, height]
            self.area_history.append(box[2] * box[3])
            self.centroid_history.append((box[0], box[1]))
            self.conf_history.append(conf)
        else:
            self.area_history.append(0)
            self.conf_history.append(0)

    def check_confidence_duration(self, threshold=0.7, duration_sec=2.0):
        """Pattern 1: Is there a stable high-confidence detection?"""
        req_frames = int(duration_sec * self.fps)
        if len(self.conf_history) < req_frames: return False
        recent = list(self.conf_history)[-req_frames:]
        return all(c >= threshold for c in recent)

    def check_growth(self, growth_ratio=1.05):
        """Pattern 2: Is the bounding box area expanding over time?"""
        if len(self.area_history) < self.fps: return False
        old_area = self.area_history[0]
        current_area = self.area_history[-1]
        if old_area == 0: return False
        return (current_area / old_area) >= growth_ratio

    def check_flicker_frequency(self):
        """Pattern 3: Counts state changes (Appearance/Disappearance)."""
        if len(self.detection_history) < self.fps: return 0.0
        h = list(self.detection_history)
        transitions = sum(1 for i in range(1, len(h)) if h[i] != h[i-1])
        return transitions / (len(h) / self.fps)

    def check_stability(self, max_drift=0.4):
        """Pattern 4: Ensures the fire isn't 'teleporting' across the frame."""
        if len(self.centroid_history) < 2: return True
        c1, c2 = self.centroid_history[-1], self.centroid_history[-2]
        dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
        return dist < max_drift # Max drift as percentage of screen width

    def get_fire_score(self):
        """Calculates a cumulative score based on all patterns."""
        score = 0
        if not self.detection_history or not self.detection_history[-1]:
            return 0

        # Weighted Logic
        if self.check_confidence_duration(threshold=0.6, duration_sec=0.5): score += 80
        if self.check_growth(1.02): score += 60
        if self.check_flicker_frequency() > 1.0: score += 90
        if not self.check_stability(0.4): score -= 5  # Heavy penalty for teleporting
        
        return max(0, score)