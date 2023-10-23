import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


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
        data = self.data["values"][start:end]
        dk_list = range(start+1, end-1)
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
        self.li = np.array([data[d[i]:d[i+1]]["values"].mean() for i in range(K)])
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
        models = [ModelK(self.data, K) for K in range(1, self.Kmax)]
        ic_list = np.array([model.ic() for model in models])
        self.optimized = models[ic_list.argmin()]
    def plot(self):
        self.optimized.plot()

class ModelK:
    def __init__(self, data, K):
        self.data = data
        self.K = K
        self.optimize()
    def optimize(self):
        d = D(self.data, self.K)
        for i in range(3):
            a = A(d, self.data, self.K)
            d.optimize(a)
        self.optimized = {"a": a, "d": d}

    def model(self):
        a = self.optimized["a"]
        d = self.optimized["d"]
        return np.concatenate([np.full(d[k+1]-d[k], a[k]) for k in range(len(a))])

    def var(self):
        return (self.data["values"]-self.model()).var(ddof=1)
    def log_likelihood(self):
        return sum([norm.logpdf(err) for err in self.data["values"]-self.model()])
    def plot(self):
        time = self.data["time"]
        plt.plot(time, self.data["values"])
        plt.plot(time, self.model())
    def ic(self):
        return self.aic()
    def aic(self):
        k = 2*self.K - 2
        return -2 * self.log_likelihood() + 2 * k