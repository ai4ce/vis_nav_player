import redis
import os

class Storage_Bot():
    def __init__(self):
        self.r = redis.Redis(host='localhost', port=6379)
        self.count = 0
    def _unique_key(self, pose : tuple[int, int, int]) -> str:
        key = f"{pose[0]}:{pose[1]}:{pose[2]}:{self.count}"
        self.count += 1
        return key
    def disk(self, pose, path="/tmp"):
        if not os.path.exists(path):
            os.makedirs(path)
        filename = self.unique_key(pose)+".jpg"
        with open(os.path.join(path, filename), "wb") as f:
            f.write(image)
    def reset(self) -> None:
        self.count = 0
    def redis(self, pose, image) -> None:
        key = self._unique_key(pose)
        try:
            is_success, buffer = cv2.imencode(".jpg", image)
            if is_success:
                self.r.set(key, buffer.tobytes())
        except Exception as e:
            pass