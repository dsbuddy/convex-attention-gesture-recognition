import serial


class Sensor:

    def __init__(self, port=None):
        self.port = port or '/dev/ttyACM0'
        self.conn = serial.Serial(port=self.port, baudrate=9600, timeout=.1)

    def read_sample(self):
        bytes = self.conn.readline()
        line = bytes.decode().strip()
        if line:
            return [float(x) for x in line.split(',')]
        else:
            return None

    def wait_for(self, test):
        while True:
            sample = self.read_sample()
            if sample and test(sample):
                return

    def read_gesture(self, max_len=None, verbose=True):
        data = []
        while max_len is None or len(data) < max_len:
            sample = self.read_sample()
            if sample:
                if verbose:
                    print(sample)
                data.append(sample)
            elif data:
                break
        if verbose:
            print()
        return data


#class PadSensor(Sensor):

#    THRESHOLD = 120 #55
#    GESTURE_LENGTH = 25 #10

#    def __init__(self, port=None):
#        super().__init__('pad', port=port)

    # def read_sample(self):
    #     return super().read_sample()[1:]

    # def is_touch(self, sample):
    #     return max(sample) > self.THRESHOLD + 1

    # def is_off(self, sample):
    #     return max(sample) < self.THRESHOLD - 1

    # def wait_for_touch(self):
    #     self.wait_for(self.is_touch)

    # def wait_for_liftoff(self):
    #     self.wait_for(self.is_off)

    # def read_gesture(self, verbose=True):
    #     if verbose:
    #         print('Waiting for liftoff...', end='\r')
    #     self.wait_for_liftoff()
    #     if verbose:
    #         print('Waiting for gesture...', end='\r')
    #     self.wait_for_touch()
    #     return super().read_gesture(max_len=self.GESTURE_LENGTH, verbose=verbose)
