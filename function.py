import random
import numpy as np

def F1(x):
    return np.sin(10 * x) * x + np.cos(2 * x) * x
## M -1 到 1 之间的矩阵
# opt  全局最优解

class Ackley():
    '''
    Ackley function
    :param var: design variable vector
    :param M: rotation matrix
    :param opt: shift vector
    :return: value
    '''
    def __init__(self):
        # self.M = np.random.uniform(-1,1,size=(50,50))
        # self.opt = np.full((1,50),42.0960)
        # self.opt = np.reshape(self.opt,-1)
        self.DNA_BOUND = [-50,50]

    def fnc(self, var):
        var = var * self.DNA_BOUND[1]
        dim = len(var)
        # var = np.dot(self.M, (var - self.opt).T).T
        sum1 = 0
        sum2 = 0
        for i in range(dim):
            sum1 = sum1 + var[i] * var[i]
            sum2 = sum2 + np.cos(2 * np.pi * var[i])
        avgsum1 = sum1 / dim
        avgsum2 = sum2 / dim

        obj = -20 * np.exp(-0.2 * np.sqrt(avgsum1)) - np.exp(avgsum2) + 20 + np.exp(1)
        return obj

class Griewank():
    '''
    GRIEWANK function
    :param var: design variable vector
    :param M: rotation matrix
    :param opt: shift vector
    :return: value
    '''
    def __init__(self):
        # self.M = np.random.uniform(-1,1,size=(50,50))
        # self.opt = np.zeros(50)
        self.DNA_BOUND = [-100, 100]

    def fnc(self, var):
        var = var * self.DNA_BOUND[1]
        dim = len(var)
        # var = np.dot(self.M, (var - self.opt).T).T
        sum1 = 0
        sum2 = 1
        for i in range(dim):
            sum1 = sum1 + var[i] * var[i]
            sum2 = sum2 * np.cos(var[i] / np.sqrt(i+1))

        obj = 1 + 1 / 4000 * sum1 - sum2
        return obj

class Rastrigin():
    '''
    Rastrigin function
    :param var: design variable vector
    :param M: rotation matrix
    :param opt: shift vector
    :return: value
    '''
    def __init__(self):
        # self.M = np.random.uniform(-1,1,size=(50,50))
        # self.opt = np.zeros(50)  #改成一串，没有行列
        self.DNA_BOUND = [-50, 50]

    def fnc(self, var):
        var = var * self.DNA_BOUND[1]
        dim = len(var)   # 维度
        # var = np.dot(self.M, (var - self.opt).T).T     # xi
        obj = 10 * dim
        for i in range(dim):
            obj = obj + (var[i] * var[i] - 10 * (np.cos(2 * np.pi * var[i])))
        return obj

class Rosenbrock():
    '''
    ROSENBROCK function
    :param var: design variable vector
    :return: value
    '''
    def __init__(self):

        self.DNA_BOUND = [-50, 50]

    def fnc(self, var):
        var = var * self.DNA_BOUND[1]
        dim = len(var)
        obj = 0
        for i in range(dim-1):
            xi = var[i]
            xnext = var[i+1]
            new = 100 * (xnext - xi * xi)**2 + (xi - 1)**2
            obj = obj + new
        return obj

class Schwefel():
    '''
    SCHWEFEL function
    :param var: design variable vector
    :return: value
    '''
    def __init__(self):

        self.DNA_BOUND = [-500, 500]
    def fnc(self, var):
        var = var * self.DNA_BOUND[1]
        dim = len(var)
        sum = 0
        for i in range(dim):
            sum += var[i]*np.sin(np.sqrt(np.abs(var[i])))

        obj = 418.9829*dim-sum
        return obj

class Sphere():
    '''
    Sphere function
    :param var: design variable vector    设计变量向量
    :param opt: shift vector    转换向量
    :return: value
    '''
    def __init__(self):
        # self.opt = np.zeros(50)
        # self.opt = self.opt.reshape(-1)
        self.DNA_BOUND = [-50, 50]

    def fnc(self, var):
        var = var * self.DNA_BOUND[1]
        # var = var - self.opt
        obj = np.dot(var, var.T)   # 向量的点积
        return obj

class Weierstrass():
    '''
    WEIERSTASS function
    :param var: design variable vector
    :param M: rotation matrix
    :param opt: shift vector
    :return: value
    '''
    def __init__(self):
        # self.M = np.random.uniform(-1, 1, size=(50, 50))
        # self.opt = np.zeros(50)
        # self.opt = self.opt.reshape(-1)
        self.DNA_BOUND = [-0.5, 0.5]


    def fnc(self, var):
        var = var * self.DNA_BOUND[1]
        D = len(var)
        # var = np.dot(self.M, (var - self.opt).T).T
        a = 0.5
        b = 3
        kmax = 20
        obj = 0
        for i in range(D):
            for k in range(kmax + 1):
                obj = obj + (a ** k) * np.cos(2 * np.pi * (b ** k) * (var[i] + 0.5))
        for k in range(kmax + 1):
            obj = obj - D * (a ** k) * np.cos(2 * np.pi * (b ** k) * 0.5)
        return obj