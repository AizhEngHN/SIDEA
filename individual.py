import numpy as np
import scipy.optimize as op

class Individual(object):   #生成个体

    def __init__(self,D_multitask):
        self.dim = D_multitask   # 维度
        # self.a = np.random.uniform(-1,1,size=(1,D_multitask))
        self.rnvec = np.random.uniform(-1,1,size=(1,D_multitask))
        # 个体初始化 均匀分布（DNA_BOUND[low,up]   dim个

        self.factorial_costs = []
        self.factorial_ranks = []
        self.scalar_fitness = []
        self.skill_factor = None  # 技能因子

