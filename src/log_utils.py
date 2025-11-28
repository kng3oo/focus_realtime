import csv
import datetime
import os


class LogManager:
    """집중도 + 자세 + EAR + 감정 로그 CSV 저장"""

    def __init__(self):
        self.log_dir = "runs"
        os.makedirs(self.log_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_dir, f"focus_log_{ts}.csv")

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time", "focus", "yaw", "pitch", "roll", "ear", "emotion"])

    def write_row(self, focus, yaw, pitch, roll, ear, emotion):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                datetime.datetime.now().isoformat(),
                focus, yaw, pitch, roll, ear, emotion
            ])

    def get_csv_path(self):
        return self.csv_path
