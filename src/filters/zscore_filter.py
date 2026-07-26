import numpy as np
from collections import deque
from src.filters.base_filter import BaseFilter

class ZScoreFilter(BaseFilter):
    def __init__(self,window_size=50,threshold=3.0):
        self.window_size=window_size
        self.threshold=threshold
        self.window=deque(maxlen=window_size)

    def fit(self,baseline_data):
        for value in baseline_data[-self.window_size]:
            self.window.append(value)

    def detect(self,value,timestamp=None):
        if len(self.window)<10:
            self.window.append(value)
            return False,0.0

        mean=np.mean(self.window)
        std=np.std(self.window)

        if std == 0:
            z_score=0.0
        else:
            z_score=(value-mean)/std

        should_escalate=abs(z_score)>self.threshold

        self.window.append(value)

        return should_escalate,abs(z_score)

    def name(self):
        return "zscore"