
class AutoNormalization:
    def __init__(self, max_value = 0.0):
        self.max_value = max_value
    
    def __call__(self, value):
        if value > self.max_value:
            self.max_value =  value
        return value / self.max_value