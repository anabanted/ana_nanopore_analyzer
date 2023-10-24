import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import square_optimize

class Event:
    def __init__(self, event_path):
        self.event = pd.read_csv(event_path, names = ["time", "raw", "fit", "filter"])
    def max_current(self):
        return self.event["raw"].max()
    def dwell_time(self, threshold):
        if self.event["raw"].max()<=threshold:
            return 0
        return self.event[self.event["raw"]>threshold].index[-1] - self.event[self.event["raw"]>threshold].index[0]
        
    def plot(self):
        self.event.plot(x="time", y="raw")
        
    def plot_opt(self, K=None, Kmax=3):
        model = self.model(K=K, Kmax=Kmax)
        model.plot()
    def _data(self):
        return self.event.rename({'raw': 'values'}, axis='columns')
    
    def model(self, K=None, Kmax=3):
        data = self._data()
        if K is None:
            model = square_optimize.Model(data, Kmax=Kmax)
        else:
            model = square_optimize.ModelK(data, K=K)
        return model
    def count_peak(self, K=None, Kmax=5):
        a = self.model(K=K, Kmax=Kmax).A
        a_array = np.array(a)
        difference = a_array - np.append(-np.inf, a_array[:-1]) 
        def compress_list(lst):
            if not lst:  # リストが空の場合は空のリストを返す
                return []

            compressed = [lst[0]]  # 最初の要素を圧縮されたリストに追加
            for item in lst[1:]:
                if item != compressed[-1]:
                    compressed.append(item)
            return compressed

        dif_is_positive = compress_list([dif > 0 for dif in difference])
        return sum(dif_is_positive)
        

class AllEvent:
    def __init__(self, event_list):
        self.event_list = event_list
    def scatter_dwelltime_maxcurrent(self, threshold, xscale='log', yscale='linear', right=None):
        dwell_times = list(map(lambda event_path: Event(event_path).dwell_time(threshold), self.event_list))
        max_current = list(map(lambda event_path: Event(event_path).max_current(), self.event_list))
        ax = plt.gca()
        ax.scatter(dwell_times, max_current)
        ax.set_xlabel('dwell_time')
        ax.set_ylabel('max_current')
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xscale=='log':
            ax.set_xlim(left=1, right=right)
        else:
            ax.set_xlim(right=right)

