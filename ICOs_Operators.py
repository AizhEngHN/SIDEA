import math
import random
import numpy as np

# from start import DNA_BOUND

def Intrication_Operator(p, dim, mum):  # 错卦算子
    # p 为genotype， ndarray类型
    # dim = p.shape[0]
    p = np.reshape(p, -1)
    p_tmp = np.copy(p)
    for i in range(dim):
        if np.random.uniform() < (1 / dim):
            u = np.random.uniform()
            if u <= 0.5:
                delta = (2 * u) ** (1 / (1 + mum)) - 1
                p_tmp[i] = p[i] + delta * p[i]
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (1 + mum))
                p_tmp[i] = p[i] + delta * (1 - p[i])
    return p_tmp


def Turnover_Operator(p, dim):  # 反转算子
    p = np.reshape(p, -1)
    H_T_O = p.copy()
    pt = random.random()  # 随机0，1之间的数,来选择翻转部位

    r = np.random.randint(1, dim)  # 从 1到 L间随机一个整数
    ######################################################################
    # r 可以自适应
    ######################################################################
    if (pt > 0.5):  # 取前 r 个进行翻转
        H_t = H_T_O[:r]
        H_T_O[:r] = np.flipud(H_t)  # 翻转函数
    else:  # 翻转后L-r 个
        H_t = H_T_O[dim - r:]
        H_T_O[dim - r:] = np.flipud(H_t)

    return H_T_O


def Mutual_Operator(p, dim,r):  # 互卦算子
    p = np.reshape(p,-1)
    H_M_O = p.copy()
    r = int(r)
    # r = round(dim / 6)  # 错卦的长度、感觉重要的部分
    #####################################################################
    #  r 可以自适应
    # 1： 随时间t 增大 Mr 变小  Mr ∈ 【0，dim/4】
    # r = round( ((Max_iteration- t) / Max_iteration)* int(dim/4)   )   # 需要参数Max_iteration 、 t
    # 2：根据适应度自适应
    # α = (Fitness - F_mean) / (F_best - F_mean)
    # if α > 1 :   # 因为 F 越小越好 ,  α >1 时 F 不好、 需要增大变化程度
        # r变大，
    # elif 0 < α < 1 #  说明F 比 F_best 好  需要减小变化程度
    #   r 变小
    # elif  α < 0  #  说明 F 比 F_mean 差 ，需要 大大增加 r
    #       r 大大变




    ######################################################################
    # F_best = min(fitness_value)
    # F_mean = np.mean(fitness_value)
    # deta = (F_best - F_mean)/ (fitness_value[j] - F_mean)
    A = p.copy()
    # print(np.shape(A))
    # print(A[12])
    for i in range(0, math.ceil(dim / 2)):
        H_M_O[i] = A[r + i]
    for i in range(math.ceil(dim / 2), dim):
        H_M_O[i] = A[i - r]

    # H_M_O = h[r:r + math.ceil(L / 2)] + h[math.ceil(L / 2) - r:L - r]
    return H_M_O


def env_factor(p, p_best, beta,low_bound,up_bound):
    aaa = beta * p + (1 - beta) * p_best
    b = np.clip(aaa,low_bound, up_bound)
    return b


def ICO_mapping(H, H_T_O, H_M_O):
    H_mapping = [H, H_T_O, H_M_O]
    return H_mapping


def fitness_function(H, function):  # 输入种群 返回该种群对应的适应度值
    fitness_value = []  # 适应度值矩阵
    for j in H:
        fitness_value.append(function.fnc(j))
    return fitness_value

# def kt_and_env(p,p_best,beta,low_bound,up_bound):
#     aa = p_best * beta + ( 1- beta ) * p
#     bb = np.clip(aa, low_bound, up_bound)
#     return bb