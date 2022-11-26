import time

class Timer:

    def __init__(self):
        self.start_time = self.time()

    def restart(self):
        self.start_time = self.time()

    def current(self):
        return self.time() - self.start_time

    @staticmethod
    def time():
        return int(round(time.time() * 1000))

    def __str__(self):
        return str(self.current())