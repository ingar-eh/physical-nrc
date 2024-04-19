'''
OPTIMIZE using scipy function --> scipy.optimize.minimize
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
import autograd.numpy as np
from autograd import jacobian
import math as mt
import itertools as it


def b(m, N, M, eta=1, Rc=0.5):
    b = m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))
    return b
# function that calculate the speed of sound underwater
def c(T=20, s=35, z=20):
    return (1449.2 + 4.6 * T - 0.055 * (T ** 2) + 0.00029 * (T ** 3) + (1.34 - 0.01 * T) * (s - 35) + 0.16 * z)

# cost function
def f(x, pc, bb, sea_cond=np.array([12, 35, 20, 300]), eta=1, B=4000, toh=1e-3, Rc=0.5, mu=0.2, tau=0.025):
    """Returns the cost function."""
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])

    return  np.log(m * (1 + pc) * N + B * (toh + td)) - \
            np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta) - np.log(np.log2(M)) + \
            - 1 / bb * np.log(-(1 - m)) - 1 / bb * np.log(-(m - 40)) \
            - 1 / bb * np.log(-(N - 2000)) - 1 / bb * np.log(-(400 - N)) \
            - 1 / bb * np.log(-(2 - M)) - 1 / bb * np.log(-(M - 64)) \
            - 1 / bb * np.log(-((m * (N / eta) * (np.log2(M)) * (0.2 * np.exp(-((3 * 100) / (2 * (M - 1))))) ** (1 / Rc)) - 0.1))

    '''np.exp((1 - m)*bb) + np.exp((m - 40) * bb) 
    + np.exp((N-2000)*bb) + np.exp((400 - N)*bb)
    + np.exp((2 - M)*bb) + np.exp((M-64)*bb) +
    np.exp(((m * (N / eta) * np.log2(M) * (0.2 * np.exp(-((3 * 100) / (2 * (M - 1))))) ** (1 / Rc)) * bb))'''

def fs(x, pc, bb):
    return f(x, pc, bb, sea_cond=np.array([16, 30, 20, 300])) +\
        f(x, pc, bb, sea_cond=np.array([11, 40, 40, 300])) +\
        f(x, pc, bb, sea_cond=np.array([12, 35, 30, 200])) + \
        f(x, pc, bb, sea_cond=np.array([10, 38, 20, 200]))+\
        f(x, pc, bb, sea_cond=np.array([9, 35, 15, 150]))
    # + f(x, pc, sea_cond=np.array([15, 30, 10, 600]))

# gradient of J0
def g(x, pc, bb, sea_cond=np.array([12, 35, 40, 300]), toh=1e-3, B=4000, eta=0.2, mu=0.2, Rc=0.5, pb=0.1):
    """Returns the gradient of the objective function.
    """
    # Helper variables
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    L = (toh + td) * B
    p = 1 + pc

    dJdN = -1 / N + ((m * p) / (L + N * m * p)) + bb * np.exp((N-2000)*bb) - bb*np.exp((400 - N)*bb)
    dJdm = -1 / m + ((N * p) / (L + N * m * p)) - (bb) * np.exp((1 - m)*bb) + bb * np.exp((m - 40) * bb)
    dJdM = -1 / (M * np.log(2) * np.log(M)) + bb * np.exp((M-64)*bb) - bb*np.exp((2 - M)*bb)
    return np.array([dJdm, dJdN, dJdM]).T

def gs(x, pc, bb):
    return jacobian(f)(x, pc, bb, sea_cond=np.array([16, 30, 20, 300])) +\
           jacobian(f)(x, pc, bb, sea_cond=np.array([11, 40, 40, 300])) + \
           jacobian(f)(x, pc, bb, sea_cond=np.array([12, 35, 30, 200])) + \
        jacobian(f)(x, pc, bb, sea_cond=np.array([10, 38, 20, 200])) + \
        jacobian(f)(x, pc, bb, sea_cond=np.array([9, 35, 15, 150]))
    # + g(x, pc, sea_cond=np.array([15, 30, 10, 600]))

# hessian of J0
def h(x, pc, bb, sea_cond=np.array([12, 35, 20, 300]), toh=1e-3, B=4000, mu=0.2, tau=0.025, eta=1, Rc=0.5, pb=0.1):
    """Returns the Hessian of the objective function.
    """
    # Helper variables
    m = x[0]
    N = x[1]
    M = x[2]
    td = sea_cond[3] / c(sea_cond[0], sea_cond[1], sea_cond[2])
    L = (toh + td) * B
    p = 1 + pc

    d2JdN2 = -((m ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + 1 / (N ** 2) \
                + (bb**2) * np.exp((N-2000)*bb) + (bb**2)*np.exp((400 - N)*bb)
    d2Jdm2 = -((N ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + 1 / m ** 2 + (bb ** 2) * np.exp((1 - m) * bb) + (bb ** 2) * np.exp((m - 40) * bb)
    d2JdmdN = -((N * m) * (p ** 2)) / ((L + N * m * p) ** 2) + p / (L + N * m * p)
    d2JdM2 = 1 / ((M ** 2) * np.log(M) * np.log(2)) + 1 / (((M) * np.log2(M) * np.log(
        2)) ** 2) + (bb ** 2) * np.exp((M - 64) * bb) + (bb ** 2) * np.exp((2 - M) * bb)

    return np.array([[d2Jdm2, d2JdmdN, 0],
                    [d2JdmdN, d2JdN2,  0],
                    [  0    ,   0,  d2JdM2]])

def hs(x, pc, bb):
    return jacobian(jacobian(f))(x, pc, bb, sea_cond=np.array([16, 30, 20, 300])) +\
           jacobian(jacobian(f))(x, pc, bb, sea_cond=np.array([11, 40, 40, 300])) +\
            jacobian(jacobian(f))(x, pc, bb, sea_cond=np.array([12, 35, 30, 200])) + \
            jacobian(jacobian(f))(x, pc, bb, sea_cond=np.array([10, 38, 20, 200])) + \
             jacobian(jacobian(f))(x, pc, bb, sea_cond=np.array([9, 35, 15, 150]))
    # + h(x, pc, sea_cond=np.array([15, 30, 10, 600]))


def BER(x):
    return np.log(x[0]) + np.log(2 ** x[1]) + np.log(np.log2(x[2])) + 2 * (
                np.log(0.2) - ((3 * 100) / (2 * (x[2] - 1))))


def centralized_solution(x0, bb):
    # callback function: save the actual cost at the k-th iteration
    def fun_update(xk):  # , opt):
        fun_evolution.append(f(xk, pc, bb))


    Nx = 64  # Non-data subcarriers
    B = 4000  # Bandwidth
    k = 2  # rel doppler margin (3) not higher than 5 or under 1
    v = 1  # doppler spread (hz)
    tau = 0.025  # delay spread (s)
    pc = 0.25  # cyclic prefix

    fun_evolution = []
    '''cons = ({'type': 'ineq', 'fun': lambda x: -(np.log(x[0]) + np.log(x[1]) + np.log(np.log2(x[2])) + 2 * (
                np.log(0.2) - ((3 * 100) / (2 * (x[2] - 1)))) - np.log(0.2))})  # prob_loss = 0.1, eta = 1, Rc=0.5'''
    res = opt.minimize(fs,
                       x0,
                       args=(pc, bb),
                       method='Newton-CG', #'trust-ncg',#
                       callback=fun_update,
                       jac=gs,
                       hess=hs,
                       tol=1e-3,
                       options={'maxiter': 15000,
                                'disp': True,
                                })

    '''
    Quantization: need quantized value. round the result and evaluate if it's better take the upper or lower value
    '''
    '''
    m_up = mt.ceil(res.x[0])
    m_low = int(res.x[0])
    M_up = mt.ceil(res.x[2])
    M_low = int(res.x[2])
    N_low = int(np.log2(res.x[1]))

    xx = [m_low, N_low, M_up]
    ber = BER(xx)
    cost_function = f([m_low, N_low, M_low], 0.25)
    if BER([m_up, N_low, M_up]) <= ber:
        ber = BER(m_up, N_low, M_up)
        xx[0] = m_up
    if BER([xx[0], N_low, M_low]) <= ber:
        ber = BER([xx[0], N_low, M_low])
        xx[2] = M_low
        cost_function = f([xx[0], N_low, xx[2]], 0.25)
    if BER([xx[0], 10, xx[2]]) <= ber:
        print('GG')
    '''

    print(f"final value of OFDM parameters:\n m:{res.x[0]}\n N:{res.x[1]}\n M:{res.x[2]}\n")
    print(f"success: {res.success}\n status:{res.message}\n")

    # evolution of the cost function
    plt.plot(range(res.nit), fun_evolution)
    plt.xlabel('Number of iterations')
    plt.ylabel('cost function')
    plt.grid()
    # plt.show()
    # plt.savefig('optim_l-bfgs-b.png')
    return res.x

'''
old derivatives (not sure about correctness)
dJdm = -1 / m + ((N * p) / (L + N * m * p)) + \
               + aa * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) * \
               np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) - \
               (bb) * np.exp((1 - m) * bb)
        dJdN = -1 / N + ((m * p) / (L + N * m * p)) + \
               + aa * (m / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) * \
               np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) + \
               bb * np.exp((N - 2000) * bb) - bb * np.exp((400 - N) * bb)
        dJdM = -1 / (M * np.log(2) * np.log(M)) + \
               + aa * m * (N / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                             + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                    -((3 * 100) / (2 * (M - 1) * Rc)))) * \
               np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) \
               + bb * np.exp((M - 64) * bb) - bb * np.exp((2 - M) * bb)
               
d2Jdm2 = -((N ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + 1 / (m ** 2) + \
                 (aa * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) ** 2) * \
                 np.exp(
                     ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) + \
                 (bb ** 2) * np.exp((1 - m) * bb)
        d2JdmdN = -((N * m) * (p ** 2)) / (L + N * m * p) ** 2 + p / (L + N * m * p) + \
                  (aa ** 2) * (m * N / (eta ** 2)) * (
                          np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) ** 2) * \
                  np.exp(
                      ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa)
        d2JdmdM = aa * (M / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                          + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                    -((3 * 100) / (2 * (M - 1) * Rc)))) * np.exp(
            ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) \
                  + (aa ** 2) * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) * \
                  m * (N / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                         + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                    -((3 * 100) / (2 * (M - 1) * Rc)))) * \
                  np.exp(
                      ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa)
        d2JdN2 = -((m ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + (1 / (N ** 2)) + \
                 (aa * (m / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) ** 2) * \
                 np.exp(
                     ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) + \
                 (bb ** 2) * np.exp((N - 2000) * bb) + (bb ** 2) * np.exp((400 - N) * bb)
        d2JdNdM = aa * (m / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                          + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                    -((3 * 100) / (2 * (M - 1) * Rc)))) * np.exp(
            ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) \
                  + (aa ** 2) * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) * \
                  m * (m / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                         + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                    -((3 * 100) / (2 * (M - 1) * Rc)))) * \
                  np.exp(
                      ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa)
        d2JdM2 = 1 / ((M ** 2) * np.log(M)) + 1 / ((M) ** 2 * np.log(M) ** 2) + \
                 aa * ((m * N * 0.2 / eta) * (
                    -1 / ((M ** 2) * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                    + 1 / (M * np.log(2)) * (3 * 100) * 1 / ((M - 1) ** 2) * np.exp(
                -((3 * 100) / (2 * (M - 1) * Rc))) - \
                    np.log2(M) * (3 * 100) * 1 / ((M - 1) ** 3) * np.exp(
                -((3 * 100) / (2 * (M - 1) * Rc))) + \
                    np.log2(M) * (((3 * 100 / 2) * 1 / ((M - 1) ** 2)) ** 2) * np.exp(
                -((3 * 100) / (2 * (M - 1) * Rc))))) \
                 * np.exp(
            ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) + \
                 (aa * m * (N / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                              + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                             -((3 * 100) / (2 * (M - 1) * Rc)))) ** 2) * \
                 np.exp(
                     ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) + \
                 + (bb ** 2) * np.exp((M - 64) * bb) + (bb ** 2) * np.exp((2 - M) * bb)

'''