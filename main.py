import numpy as np
import ICOs_Operators
import geatpy as ea
from individual import Individual
from task import Task
import math
import scipy.stats
import matplotlib.pyplot as plt

reps = 20  # 重复次数
pop = 100  # 种群个体数
Max_iteration = 500  # 最大迭代次数
# Pm_I = 0.1  # 错卦算子 突变概率 小于Pm_I突变
Pop = []  # 种群
T = 0  # 迭代次数
# P_best = []  # 最佳个体
# F_best = []  # 最佳个体适应度
Tasks = ['CI_HS', 'CI_MS', 'CI_LS', 'PI_HS', 'PI_MS', 'PI_LS', 'NI_HS', 'NI_MS', 'NI_LS']
Dim = 50
FFF =np.ones((2, reps)) # 二十次的最优解

mum = 10

for tasks in Tasks:
# if 1>0:
#     tasks = 'CI_LS'
    for rep in range(reps):
        print('Repetition: ' + str(rep) + ' :')

        Tasks = Task(tasks)
        no_of_tasks = len(Tasks)
        # D = np.zeros(shape=no_of_tasks)  # 维度（建立一个 1，no_of_tasks 的 全零向量）
        P_best = []
        F_best = np.ones((no_of_tasks, Max_iteration))
        D_D = np.zeros((no_of_tasks, Max_iteration))
        # D_D[0][0] = 1000
        # D_D[1][0] = 1000
        S_S = []
        miu = np.ones((no_of_tasks, Max_iteration))
        r = np.zeros(no_of_tasks)
        r_kt = np.zeros(no_of_tasks)

        for i in range(no_of_tasks):
            P_best.append(9e9999)  # 最佳个体
            F_best[0][0] = 9e9999  # 最佳个体适应度
            F_best[0][-1] = 9e9999  # 最佳个体适应度
            F_best[1][0] = 9e9999  # 最佳个体适应度
            F_best[1][-1] = 9e9999  # 最佳个体适应度
            S_S.append(0)  # S_S[i] = 1 是 F_best(t-1)<F_best(t)

        population = np.asarray([Individual(Dim)
                                 for _ in range(pop)])

        for t in range(Max_iteration):

            for j in range(pop):
                population[j].factorial_costs = []
                population[j].factorial_ranks = []
                population[j].scalar_fitness = []
                population[j].skill_factor = None

            New_population = []
            fitness_value = []

            beta = (Max_iteration - t) / Max_iteration * 0.9
            number = 0
            for i in range(no_of_tasks):
                for j in range(pop):
                    fitness_value.append(ICOs_Operators.fitness_function(population[j].rnvec, Tasks[i]))  # 适应度矩阵
                    population[j].factorial_costs.append(fitness_value[number])  # 因子花费
                    # print(population[j].factorial_costs)
                    number = number + 1
                if F_best[i][t - 1] > min(min(fitness_value[i * pop:pop * (i + 1)])):
                    F_best[i][t] = min(min(fitness_value[i * pop:pop * (i + 1)]))
                    S_S[i] = 1
                else:
                    F_best[i][t] = min(min(fitness_value[i * pop:pop * (i + 1)]))
                    S_S[i] = 0

                P_best[i] = population[fitness_value[i * pop:pop * (i + 1)].index(
                    min(fitness_value[i * pop:pop * (i + 1)]))].rnvec  # 每个任务的最佳个体
                b = np.reshape(fitness_value, -1)
                rank = b[(i * pop):(pop * (i + 1))].argsort()   # 当前任务适应度升序排名


                for j in range(pop):
                    population[rank[j]].factorial_ranks.append(j+1)  # 因子排名

            for i in range(pop):
                population[i].scalar_fitness = 1 / np.min(population[i].factorial_ranks)  # 标量适应度
                if population[i].factorial_ranks[0] == population[i].factorial_ranks[1]:   # 技能因子赋值
                    population[i].skill_factor = np.random.randint(0,2,1)
                else:
                    population[i].skill_factor = np.argmin(population[i].factorial_ranks)

            HH_mapping = []
            New_population = []
            subpop = [[], []]
            for j in range(no_of_tasks):  # 计算r 为知识转移准备
                k = np.zeros(no_of_tasks)  # 子群数量
                sum = 0

                for i in range(pop):

                    if population[i].skill_factor == j:
                        subpop[j].append(population[i].rnvec)

                D_D[j][t] = np.std(subpop[j])  # 求得多样性
                miu[j][t] = math.tanh(D_D[j][t])

                if t > 1:
                    if t == 1:
                        aaa = 1
                    else:
                        aaa = 0
                    if S_S[j] == 1 or (D_D[j][t] / t) <= (D_D[j][t - 1] / ((t - 1) + aaa)):
                        miu[j][t] = miu[j][t - 1]
                    elif (D_D[j][t] / t) > (D_D[j][t - 1] / (t - 1)):
                        miu[j][t] = math.tanh(D_D[j][t])

                r[j] = miu[j][t] * round(((Max_iteration - t) / Max_iteration) * int(Dim / 4))

            if len(subpop[0]) > len(subpop[1]):
                subpop[0] = subpop[0][:len(subpop[1])]
            else:
                subpop[1] = subpop[1][:len(subpop[0])]
            # 知识转移：KL散度判断

            KL = scipy.stats.entropy(subpop[0], subpop[1])
            KL = np.mean(KL)

            r_kt[0] = round((1 - (1 / (KL + 1))) * r[0] + (1 / (KL + 1)) * r[1])
            r_kt[1] = round((1 - (1 / (KL + 1))) * r[1] + (1 / (KL + 1)) * r[0])

            for j in range(no_of_tasks):
                H_mapping = []
                num_subpop = 0
                for i in range(pop):

                    if population[i].skill_factor == j:
                        num_subpop = num_subpop + 1
                        # 原本的种群
                        H_mapping.append(population[i].rnvec)
                        # 突变后的种群
                        xxx = ICOs_Operators.Intrication_Operator(
                                population[i].rnvec, Dim, mum
                            )
                        xxx = np.reshape(xxx,(1,50))

                        H_mapping.append(xxx)
                        #M
                        x = ICOs_Operators.Mutual_Operator(
                            population[i].rnvec, Dim, r_kt[j]
                                )
                        x = np.reshape(x, (1, 50))
                        H_mapping.append(x)
                        # MT
                        xx = ICOs_Operators.Mutual_Operator(
                                    ICOs_Operators.Turnover_Operator(
                                        population[i].rnvec, Dim
                                    ), Dim, r_kt[j]
                                )
                        xx = np.reshape(xx,(1,50))
                        H_mapping.append(xx)
                        #ET
                        H_mapping.append(ICOs_Operators.env_factor(
                                ICOs_Operators.Turnover_Operator(
                                    population[i].rnvec, Dim
                                )
                            , P_best[j], beta, -1, 1
                        ))

                H_mapping = np.reshape(H_mapping, (5 * num_subpop, Dim))
                # print(np.shape(H_mapping))
                fitness_value = ICOs_Operators.fitness_function(H_mapping, Tasks[j])
                fitness_value_matrix = np.reshape(np.array(fitness_value), (5 * num_subpop, 1))
                if num_subpop > 0:
                    New_Pop_Index = ea.etour(fitness_value_matrix * (-1), num_subpop)

                    for m in New_Pop_Index:
                        New_population.append(H_mapping[m])  # 循环到最后 New_popu里有pop个种群的 rnvec 值

            # 更新种群类

            for i in range(pop):
                # print(np.shape(New_population))
                population[i].rnvec = New_population[i]
                population[i].rnvec = np.reshape(population[i].rnvec, (1, Dim))

        print(F_best[0][Max_iteration - 1])
        print(F_best[1][Max_iteration - 1])
        # print(F_best[0])
        x = np.linspace(1, Max_iteration, Max_iteration)
        x2 = np.linspace(1, Max_iteration, Max_iteration)
        y = F_best[0]
        y2 = F_best[1]
        fig1 = plt.figure(1)
        plt.plot(x, y, ls="-", lw=2, label="S-IDEA")
        plt.xlabel('GENERATIONS')
        plt.ylabel('OBJECTIVE')
        plt.legend()
        plt.savefig('./{}/{}_T1实验图{}.jpg'.format(tasks, tasks, rep))
        plt.close(fig1)

        fig2 = plt.figure(2)
        plt.plot(x2, y2, ls="-", lw=2, label="S-IDEA")
        plt.xlabel('GENERATIONS')
        plt.ylabel('OBJECTIVE')
        plt.legend()
        plt.savefig('./{}/{}_T2实验图{}.jpg'.format(tasks, tasks, rep))
        plt.close(fig2)
        np.savetxt("./{}/{}_result{}.txt".format(tasks, tasks, rep), F_best)
        FFF[0][rep] = F_best[0][Max_iteration - 1]
        FFF[1][rep] = F_best[1][Max_iteration - 1]
    print(FFF[0])
    print(FFF[1])
    print("{}:".format(tasks),"T1:{}".format(np.mean(FFF[0])),"(",np.std(FFF[0]),")"
          ,"   ","T2:{}".format(np.mean(FFF[1])),"(",np.std(FFF[1]),")")


        # plt.show()
