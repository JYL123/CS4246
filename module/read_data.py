import pickle
import os

class read:
    def __init__(self, action_value, action_times, value_path, times_path):
        self.action_value = action_value
        self.action_times = action_times
        self.value_path = value_path
        self.times_path = times_path

        value_file_size = os.stat(self.value_path).st_size
        if value_file_size != 0:
            with open(self.value_path, 'rb') as handle:
                self.action_value = pickle.loads(handle.read())

        times_file_size = os.stat(self.times_path).st_size
        if times_file_size != 0:
            with open(self.times_path, 'rb') as handle:
                self.action_times = pickle.loads(handle.read())
