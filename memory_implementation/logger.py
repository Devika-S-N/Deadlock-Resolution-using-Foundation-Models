# logger.py

import os
import threading
from datetime import datetime

class Logger:
    def __init__(self, log_dir="logs", strict_order=True):
        os.makedirs(log_dir, exist_ok=True)
        self.timestamp = datetime.now().strftime('%H:%M:%S_%d-%m')
        self.log_path = os.path.join(log_dir, f"log_{self.timestamp}.txt")
        # line-buffered text file; encoding for safety
        self._fh = open(self.log_path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()
        self._strict = bool(strict_order)

    def log(self, message: str):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{ts}] {message}\n"
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            if self._strict:
                try:
                    os.fsync(self._fh.fileno())
                except OSError:
                    pass  # fsync may be unavailable on some filesystems

    def get_timestamp(self):
        return self.timestamp

    def close(self):
        with self._lock:
            try:
                self._fh.flush()
                if self._strict:
                    os.fsync(self._fh.fileno())
            finally:
                self._fh.close()
