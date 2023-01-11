import numpy as np
from function import  Ackley, Griewank, Rastrigin, Rosenbrock, \
                                Schwefel, Sphere, Weierstrass

def Task(name):
    if name == 'CI_HS':
        Tasks = [Griewank(), Rastrigin()]
        return Tasks

    elif name == 'CI_MS':
        Tasks = [Ackley(), Rastrigin()]
        return Tasks
    elif name == 'CI_LS':
        Tasks = [Ackley(), Schwefel()]
        return Tasks
    elif name == 'PI_HS':
        Tasks = [Rastrigin(), Sphere()]
        return Tasks
    elif name == 'PI_MS':
        Tasks = [Ackley(), Rosenbrock()]
        return Tasks
    elif name == 'PI_LS':
        Tasks = [Ackley(), Weierstrass()]
        return Tasks
    elif name == 'NI_HS':
        Tasks = [Rosenbrock(), Rastrigin()]
        return Tasks
    elif name == 'NI_MS':
        Tasks = [Griewank(), Weierstrass()]
        return Tasks
    elif name == 'NI_LS':
        Tasks = [Rastrigin(), Schwefel()]
        return Tasks

