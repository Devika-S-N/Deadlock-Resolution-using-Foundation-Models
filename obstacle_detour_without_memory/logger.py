# logger.py

import os
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%H:%M:%S_%d-%m')
        self.log_path = os.path.join(self.log_dir, f"log_{self.timestamp}.txt")
        self.file = open(self.log_path, "w")

    def log(self, message):
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.file.write(f"[{time_str}] {message}\n")

    def get_timestamp(self):
        return self.timestamp

    def close(self):
        self.file.close()
