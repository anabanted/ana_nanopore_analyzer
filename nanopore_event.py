import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from . import square_optimize


class Event:
    def __init__(self, event_path):
        self.event = pd.read_csv(
            event_path, names=["time", "raw", "fit", "filter"])

    def max_current(self):
        return self.event["raw"].max()

    def baseline(self):
        return square_optimize.EventFinder(self._data()).baseline()

    def max_blockage(self):
        return self.max_current() - self.baseline()

    def _dwell_time(self, threshold):
        if self.event["raw"].max() <= threshold:
            return 0
        return (
            self.event[self.event["raw"] > threshold].index[-1]
            - self.event[self.event["raw"] > threshold].index[0]
        )

    def dwell_time(self):
        return square_optimize.EventFinder(self._data()).dwell_time()

    def plot(self):
        ax = self.event.plot(x="time", y="raw")
        ax.set_xlabel("time /us")
        ax.set_ylabel("current /pA")
        plt.show()

    def plot_opt(self, K=None, Kmax=3):
        model = self.model(K=K, Kmax=Kmax)
        model.plot()

    def _data(self):
        return self.event.rename({"raw": "values"}, axis="columns")

    def model(self, K=None, Kmax=3):
        data = self._data()
        if K is None:
            model = square_optimize.Model(data, Kmax=Kmax)
        else:
            model = square_optimize.ModelKTotalData(data, K=K)
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
    def __init__(self, event_list, Kmax=5):
        self.event_list = event_list
        self.Kmax = Kmax

    @staticmethod
    def _safe_dwell_time(event_path):
        try:
            return Event(event_path).dwell_time()
        except:
            return None

    @staticmethod
    def _safe_max_current(event_path):
        try:
            return Event(event_path).max_current()
        except:
            return None

    @staticmethod
    def _safe_baseline(event_path):
        try:
            return Event(event_path).baseline()
        except:
            return None

    @staticmethod
    def _safe_max_blockage(event_path):
        try:
            return Event(event_path).max_blockage()
        except:
            return None

    @staticmethod
    def _safe_peak_count(event_path, K=None, Kmax=5):
        try:
            return Event(event_path).count_peak(K=K, Kmax=Kmax)
        except:
            return None

    def dwell_times(self):
        return [
            self._safe_dwell_time(event_path)
            for event_path in self.event_list
            if self._safe_dwell_time(event_path) is not None
        ]

    def max_currents(self):
        return [
            self._safe_max_current(event_path)
            for event_path in self.event_list
            if self._safe_max_current(event_path) is not None
        ]

    def baselines(self):
        return [
            self._safe_baseline(event_path)
            for event_path in self.event_list
            if self._safe_baseline(event_path) is not None
        ]

    def max_blockages(self):
        return [
            self._safe_max_blockage(event_path)
            for event_path in self.event_list
            if self._safe_max_blockage(event_path) is not None
        ]

    def peak_counts(self, K=None):
        Kmax = self.Kmax
        return [
            self._safe_peak_count(event_path, K=K, Kmax=Kmax)
            for event_path in self.event_list
            if self._safe_peak_count(event_path, K=K, Kmax=Kmax) is not None
        ]

    def scatter_dwelltime_maxcurrent(self, xscale="log", yscale="linear", right=None):
        dwell_times = self.dwell_times()
        max_current = self.max_currents()
        ax = plt.gca()
        ax.scatter(dwell_times, max_current)
        ax.set_xlabel("dwell_time")
        ax.set_ylabel("max_current")
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        if xscale == "log":
            ax.set_xlim(left=1, right=right)
        else:
            ax.set_xlim(right=right)
