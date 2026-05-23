import os


class Tee(object):
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        if not os.path.exists(logfile):
            open(logfile, 'w').close()
        self.log = open(logfile, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()
