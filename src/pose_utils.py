import cv2
import numpy as np
import mediapipe as mp

class HeadPoseEstimator:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

        self.idx = [1,152,33,263,61,291]

        self.model_3d = np.array([
            [0,0,0],
            [0,-63.6,-12.5],
            [-43.3,32.7,-26],
            [43.3,32.7,-26],
            [-28.9,-28.9,-24.1],
            [28.9,-28.9,-24.1],
        ], dtype=np.float32)

    def _euler(self, R):
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy >= 1e-6:
            pitch = np.degrees(np.arctan2(R[2,1], R[2,2]))
            yaw   = np.degrees(np.arctan2(-R[2,0], sy))
            roll  = np.degrees(np.arctan2(R[1,0], R[0,0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))
            yaw   = np.degrees(np.arctan2(-R[2,0], sy))
            roll  = 0
        return yaw, pitch, roll

    def score(self, frame):
        h, w = frame.shape[:2]
        res = self.face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return 0.5, (0,0,0)

        lm = res.multi_face_landmarks[0].landmark
        pts_2d = np.array([[lm[i].x*w, lm[i].y*h] for i in self.idx], dtype=np.float32)

        f = max(w, h)
        cam = np.array([[f,0,w/2],[0,f,h/2],[0,0,1]], dtype=np.float32)
        dist = np.zeros((4,1))

        ok, rvec, tvec = cv2.solvePnP(self.model_3d, pts_2d, cam, dist)
        if not ok:
            return 0.5, (0,0,0)

        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, roll = self._euler(R)

        score = 1 - (abs(yaw)/25 + abs(pitch)/20 + abs(roll)/20)/3
        score = float(np.clip(score, 0, 1))

        return score, (yaw, pitch, roll)
