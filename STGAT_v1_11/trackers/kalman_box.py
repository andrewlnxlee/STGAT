import numpy as np

class KalmanBoxTracker:
    count = 0
    def __init__(self, centroid):
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.x = np.array([centroid[0], centroid[1], 0, 0])
        self.P = np.eye(4) * 10.0
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, measurement):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        z = np.array(measurement)
        y = z - self.H @ self.x
        R = np.eye(2) * 5.0
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def predict(self):
        Q = np.eye(4) * 1.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.x[:2]