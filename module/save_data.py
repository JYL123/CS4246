import pickle
import os

class save:
    def __init__(self, action_value, action_times, value_path, times_path):
        self.action_value = action_value
        self.action_times = action_times
        self.value_path = value_path
        self.times_path = times_path

        with open(self.value_path, 'wb') as handle:
            pickle.dump(self.action_value, handle, protocol=2)
        with open(self.times_path, 'wb') as handle:
            pickle.dump(action_times, handle, protocol=2)