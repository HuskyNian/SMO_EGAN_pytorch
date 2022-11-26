import os 

class Logger:

    def __init__(self, path, header = None):
        if os.path.exists(path):
            os.remove(path)
        self.file = open(path, "wb")
        if header is not None:
            self.writeln(header)

    def write(self, *args):
        self.file.write(*args)
        self.file.flush()

    def writeln(self, *args):
        self.file.write(*args)
        self.file.write("\n".encode())
        self.file.flush()
    
    def __del__(self):
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()
            self.file = None