import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math


class D:
    def __init__(self, data, K):
        self.data = data
        self.K = K
        self.li = np.array([int(i*len(data)/K) for i in range(K+1)])
    def optimize(self, a: "A"):
        d_list = np.array([self.optimize_k(a, k) for k in range(self.K+1)])
        d_list.sort()
        self.li = d_list

    def optimize_k(self, a: "A", k):
        if k==0:
            return 0
        if k==self.K:
            return len(self.data)
        start = self[k-1]
        end = self[k+1]
        if start+1>=end:
            return self[k]
        data = self.data[start:end]
        dk_list = range(start+1, end)
        error_list = np.array([self.error(data, self.step(a[k-1], a[k], start, end, dk)) for dk in dk_list])
        dk = dk_list[error_list.argmin()]
        return dk

    @staticmethod
    def step(a1, a2, start, end, dk):
        np.full(end - dk, a2)
        return np.concatenate([np.full(dk - start,a1),np.full(end - dk,a2)])
    @staticmethod
    def error(data, model):
        return np.linalg.norm(data-model)

    def __getitem__(self, item):
        return self.li[item]
    def __str__(self):
        return str(self.li)

class A:
    def __init__(self, d: D, data, K):
        self.li = np.array([self.mean(data[d[i]:d[i+1]], data[d[i]]) for i in range(K)])
    @staticmethod
    def mean(data, di):
        if len(data)==0:
            return di
        else:
            return np.mean(data)
    def __getitem__(self, item):
        return self.li[item]
    def __len__(self):
        return len(self.li)
    def __str__(self):
        return str(self.li)

class Model:
    def __init__(self, data, Kmax=6):
        self.data = data
        self.Kmax = Kmax
        self.optimize()
    def optimize(self):
        models = [ModelKTotalData(self.data, K) for K in range(1, self.Kmax+1)]
        ic_list = np.array([model.ic() for model in models])
        self.optimized = models[ic_list.argmin()]
    def plot(self):
        self.optimized.plot()
    @property
    def A(self):
        return self.optimized.A
    @property
    def D(self):
        return self.optimized.D
class ModelK:
    def __init__(self, data, K):
        self.data = data
        self.data_ndarray = np.array(data["values"])
        self.K = K
        self.optimize()

    def optimize(self):
        d = D(self.data_ndarray, self.K)
        for i in range(3):
            a = A(d, self.data_ndarray, self.K)
            d.optimize(a)
        self.optimized = {"a": a, "d": d}

    def model(self):
        a = self.optimized["a"]
        d = self.optimized["d"]
        return np.concatenate([np.full(d[k+1]-d[k], a[k]) for k in range(len(a))])
    
    @property
    def A(self):
        return self.optimized["a"]
    @property
    def D(self):
        return self.optimized["d"]

    def var(self):
        return (self.data_ndarray-self.model()).var(ddof=1)

    def single_plateu(self, i):
        return self.data[self.optimized["d"][i]:self.optimized["d"][i+1]]

    def log_likelihood(self):
        error = self.data_ndarray-self.model()
        scale = error.std(ddof=1)
        if scale==0:
            return 0
        return sum([norm.logpdf(err, scale=scale) for err in error])
    def plot(self):
        time = self.data["time"]
        plt.plot(time, self.data["values"], label="raw")
        plt.plot(time, self.model(), label="model")
        plt.legend()
        plt.show()
        
    def ic(self):
        return self.bic()
    def aic(self):
        k = 2*self.K - 2
        return -2 * self.log_likelihood() + 2 * k
    def bic(self):
        k = 2*self.K - 2
        n = len(self.data)
        return -2 * self.log_likelihood() + k * math.log(n)
        

class EventFinder:
    def __init__(self, data):
        self.data = data
        self.model = ModelK(data, 3)
    def event(self):
        return self.model.single_plateu(1)
    def before_event(self):
        return self.model.single_plateu(0)
    def after_event(self):
        return self.model.single_plateu(2)
    def dwell_time(self):
        time = self.event()["time"]
        if len(time)==0:
            return 0
        return time.iloc[-1] - time.iloc[0]
        

class ModelKTotalData:
    def __init__(self, data, K):
        self.data = data
        ef = EventFinder(data)
        self.event = ModelK(ef.event(), K)
        self.before_event = ModelK(ef.before_event(), 1)
        self.after_event = ModelK(ef.after_event(), 1)
    def plot(self):
        time = self.data["time"]
        plt.plot(time, self.data["values"], label="raw")
        plt.plot(time, self.model(), label="model")
        plt.legend()
        plt.show()
    def ic(self):
        return self.event.ic()
    def model(self):
        return np.concatenate([self.before_event.model(), self.event.model(), self.after_event.model()])
    @property
    def A(self):
        return self.event.A
    @property
    def D(self):
        return self.event.D
