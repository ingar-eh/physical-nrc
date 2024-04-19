import numpy
import random

import numpy as np
import autograd.numpy as np
from autograd import jacobian

if False:
    class SimulationFunctionXTX_BTX:
        """This class representing the simulation function. This class can be used to calculate the gradient and hessian of the mentioned function. """

        @staticmethod
        def get_fn(x: np.array, pc=0.25, sea_cond=np.array([12, 40, 20, 300]), B=4000, toh=10e-3, eta=1,
                   Rc=0.5) -> np.array:
            """This method can be used to calculate the outcome of the function for each given Xi and Bi. X = [n, N, M, pc]"""
            m = x[0]
            N = x[1]
            M = x[2]
            aa = 50
            bb = 0.1
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                + (1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            bterm = (np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa)) + \
                    np.exp((1 - m) * bb) + \
                    np.exp((400 - N) * bb) + np.exp((N - 2000) * bb) + \
                    np.exp(((2 - M) * bb)) + np.exp((M - 64) * bb)
            f = ( np.log(m * (1 + pc) * N + B * (toh + td))
                 - np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta)
                 - np.log(np.log2(M)))
            return f

        @staticmethod
        def get_gradient_fn(x: np.array, pc=0.25, sea_cond=np.array([12, 40, 20, 300]), toh=1e-3, B=4000, eta=1, Rc=0.5,
                            pb=0.1) -> np.array:
            """This method can be used to calculate the gradient for any given Xi."""
            m = x[0]
            N = x[1]
            M = x[2]
            p = 1 + pc
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                + (1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            L = (toh + td) * B
            aa = 50
            bb = 0.1
            dJdm = -1 / m + ((N * p) / (L + N * m * p)) +\
                   + aa * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) * \
                   np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa) - \
                    (bb) * np.exp((1-m) * bb)
            dJdN = -1 / N + ((m * p) / (L + N * m * p)) +\
                   + aa * (m / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) * \
                   np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa) + \
                    bb * np.exp((N - 2000) * bb) - bb*np.exp((400-N)*bb)
            dJdM = -1 / (M * np.log(M)) +\
                   + aa * m  * (N/eta) * 0.2 * ( 1/(M*np.log(2))*np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                   + np.log2(M) * ((3*100)/2) * 1/((M-1)**2) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) ) * \
                   np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa) \
                   + bb * np.exp((M - 64) * bb) - bb*np.exp((2-M) * bb)

            '''
            dJdm = -1 / m + ((N * p) / (L + N * m * p)) + \
                   (aa * N * np.exp((aa * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                           5 * eta * np.log(2)) - aa * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                           5 * eta * np.log(2)) - (bb) * np.exp((1-m) * bb)
            dJdN = -1 / N + ((m * p) / (L + N * m * p)) + \
                   (aa * m * np.exp((aa * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                           5 * eta * np.log(2)) - aa * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                           5 * eta * np.log(2)) + bb * np.exp((N - 2000) * bb) - bb*np.exp((400-N)*bb)
            dJdM = -1 / (M * np.log(M)) + \
                   aa * np.exp(
                (aa * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (5 * eta * np.log(2)) - aa * pb) * (
                           (N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (5 * M * eta * np.log(2)) + (
                           120 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                   Rc * eta * np.log(2) * (2 * M - 2) ** 2)) + bb * np.exp((M - 64) * bb) - bb*np.exp((2-M) * bb)
            '''

            return np.array([dJdm, dJdN, dJdM])

        @staticmethod
        def get_hessian_fn(x: np.array, pc=0.25, *, sea_cond=np.array([12, 40, 20, 300]), toh=1e-3, B=4000, Rc=0.5, eta=1,
                           pb=1) -> np.array:
            """This method can be used to calculate the hessian for any given Xi."""
            m = x[0]
            N = x[1]
            M = x[2]
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                + (1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            L = (toh + td) * B
            p = 1 + pc
            aa = 50
            bb = 0.1
            d2Jdm2 = -((N ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + 1 / (m ** 2) +\
                      ( aa * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))**2) * \
                        np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa) + \
                         ((bb)**2)  * np.exp((1-m) * bb)
            d2JdmdN = -((N * m) * (p ** 2)) / (L + N * m * p) ** 2 + p / (L + N * m * p)+\
                      (aa**2) * (m * N / (eta**2)) * (np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))**2) * \
                        np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa)
            d2JdmdM = aa * (M / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                              + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                        -((3 * 100) / (2 * (M - 1) * Rc)))) * np.exp(
                ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) \
                      + (aa ** 2) * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) * \
                      m * (N / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                             + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                        -((3 * 100) / (2 * (M - 1) * Rc)))) * \
                      np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa)
            d2JdN2 = -((m ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + (1 / (N ** 2)) +\
                     (aa * (m / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))**2) * \
                     np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa) + \
                     (bb**2) * np.exp((N - 2000) * bb) + (bb**2) * np.exp((400-N)*bb)
            d2JdNdM = aa * (m / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                              + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                        -((3 * 100) / (2 * (M - 1) * Rc)))) * np.exp(
                ((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa) \
                      + (aa ** 2) * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) * \
                      m * (m / eta) * 0.2 * (1 / (M * np.log(2)) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                                             + np.log2(M) * ((3 * 100) / 2) * 1 / ((M - 1) ** 2) * np.exp(
                        -((3 * 100) / (2 * (M - 1) * Rc)))) * \
                      np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - 0.1)) * aa)
            d2JdM2 = 1 / ((M ** 2) * np.log(M)) + 1 / ((M) ** 2 * np.log(M) ** 2) +\
                     aa * ((m*N*0.2/eta)*(-1/((M**2)*np.log(2))*np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                      + 1/(M*np.log(2))*(3*100) * 1/((M-1)**2) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) - \
                      np.log2(M) * (3*100) * 1/((M-1)**3) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) + \
                      np.log2(M) * (((3*100/2) * 1/((M-1)**2))**2) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) ))\
                      * np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa) +\
                      (aa * m * (N/eta) * 0.2 * ( 1/(M*np.log(2))*np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) \
                       + np.log2(M) * ((3*100)/2) * 1/((M-1)**2) * np.exp(-((3 * 100) / (2 * (M - 1) * Rc))) )**2) * \
                       np.exp(((m * (N / eta) * np.log2(M) * 0.2 * np.exp(-((3 * 100) / (2 * (M - 1) * Rc)))-0.1)) * aa) + \
                       + (bb**2) * np.exp((M - 64) * bb) + (bb**2)*np.exp((2-M) * bb)


            '''       
            d2Jdm2 = -((N ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + 1 / (m ** 2) + \
                     ((aa ** 2) * (N ** 2) * np.exp((aa * N * m * np.exp(-300 / (Rc * 2 * (M - 1))) * np.log(M)) / (
                             5 * eta * np.log(2)) - aa * pb) * np.exp(-600 / (Rc * 2 * (M - 1))) * (
                              np.log(M) ** 2)) / ((25 * eta ** 2) * (np.log(2) ** 2)) + (bb ** 2) * np.exp((1-m)* bb)
            d2JdmdN = -((N * m) * (p ** 2)) / (L + N * m * p) ** 2 + p / (L + N * m * p) + \
                      (aa * np.exp((aa * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                              5 * eta * np.log(2)) - aa * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                              eta * np.log(2)) + \
                      (aa ** 2 * N * m * np.exp((aa * N * m * np.exp(-300 / (Rc * 2 * (M - 1))) * np.log(M)) / (
                              5 * eta * np.log(2)) - 100 * pb) * np.exp(-600 / (Rc * 2 * (M - 1))) * (
                               np.log(M) ** 2)) / (25 * (eta ** 2) * (np.log(2) ** 2))
            d2JdmdM = (20 * N * np.exp(
                (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (eta * np.log(2)) - 100 * pb) * np.exp(
                -300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) + \
                      (20 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                              eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M) * (
                               (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) +
                               (12000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                       Rc * eta * np.log(2) * (2 * M - 2) ** 2))) / (eta * np.log(2)) + (
                              12000 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                              eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                              Rc * eta * np.log(2) * (2 * M - 2) ** 2)
            d2JdN2 = -((m ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + (1 / (N ** 2)) + \
                     ((aa ** 2) * (m ** 2) * np.exp((aa * N * m * np.exp(-300 / (Rc * 2 * (M - 1))) * np.log(M)) / (
                             5 * eta * np.log(2)) - aa * pb) * np.exp(-600 / (Rc * 2 * (M - 1))) * (
                              np.log(M) ** 2)) / (25 * (eta ** 2) * (np.log(2) ** 2))\
                     + (bb ** 2) * np.exp((N - 2000) * bb) + (bb**2) * np.exp((400-N)*bb)
            d2JdNdM = (20 * m * np.exp(
                (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (eta * np.log(2)) - 100 * pb) * np.exp(
                -300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) + \
                      (20 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                              eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M) * (
                               (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (M * eta * np.log(2)) +
                               (12000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                       Rc * eta * np.log(2) * (2 * M - 2) ** 2))) / (eta * np.log(2)) + (
                              12000 * N * np.exp((20 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                              eta * np.log(2)) - 100 * pb) * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                              Rc * eta * np.log(2) * (2 * M - 2) ** 2)
            d2JdM2 = 1 / ((M ** 2) * np.log(M)) + 1 / ((M) ** 2 * np.log(M) ** 2) + \
                     (aa ** 2) * np.exp(
                (aa * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (5 * eta * np.log(2)) - aa * pb) * (
                             (aa * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (5 * M * eta * np.log(2)) + (
                             120 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                     Rc * eta * np.log(2) * (2 * M - 2) ** 2)) ** 2 \
                     - aa * np.exp(
                (aa * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (5 * eta * np.log(2)) - aa * pb) * (
                             (20 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (5 * M ** 2 * eta * np.log(2)) - (
                             240 * N * m * np.exp(-300 / (Rc * (2 * M - 2)))) / (
                                     M * Rc * eta * np.log(2) * (2 * M - 2) ** 2) + (
                                     480 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                     Rc * eta * np.log(2) * (2 * M - 2) ** 3) - (
                                     72000 * N * m * np.exp(-300 / (Rc * (2 * M - 2))) * np.log(M)) / (
                                     Rc ** 2 * eta * np.log(2) * (2 * M - 2) ** 4)) \
                             + (bb ** 2) * np.exp((M - 64)* bb) + (bb**2) * np.exp((2-M)*bb)
            '''
            return np.array([[d2Jdm2, d2JdmdN, d2JdmdM],
                             [d2JdmdN, d2JdN2, d2JdNdM],
                             [d2JdmdM,    d2JdNdM, d2JdM2]])
else:
    class SimulationFunctionXTX_BTX:
        """This class representing the simulation function. This class can be used to calculate the gradient and hessian of the mentioned function. """

        @staticmethod
        def get_fn(x: np.array, bb, pc=0.25, sea_cond=np.array([12, 40, 20, 300]), B=4000, toh=10e-3, eta=1,
                   Rc=0.5) -> np.array:
            """This method can be used to calculate the outcome of the function for each given Xi and Bi. X = [n, N, M, pc]"""
            m = x[0]
            N = x[1]
            M = x[2]

            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                + (1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])

            f = (np.log(m*(1 + pc)*N + B*(toh+td)) - np.log(m) - np.log(Rc) - np.log(B) - np.log(N / eta) - np.log(np.log2(M))) \
                - 1 / bb * np.log(-(1 - m)) - 1 / bb * np.log(-(m - 40)) - 1 / bb * np.log(
                -(N - 2000)) - 1 / bb * np.log(-(400 - N)) \
                - 1 / bb * np.log(-(2 - M)) - 1 / bb * np.log(-(M - 64)) \
                - 1 / bb *np.log(-((m * (N / eta) * (np.log2(M)) * (0.2 * np.exp(-((3 * 100) / (2 * (M - 1))))) ** (1 / Rc)) - 0.1))
            return f
            '''+ np.exp((1 - m) * bb) + np.exp((m - 40) * bb) + np.exp((N - 2000) * bb) + np.exp((400 - N) * bb) +
                np.exp((2 - M) * bb) + np.exp((M - 64) * bb) + 
                np.exp((m * (N / eta) * np.log2(M) * (0.2 * np.exp(-((3 * 100) / (2 * (M - 1))))) ** (1 / Rc)) * bb)'''


        @staticmethod
        def get_gradient_fn(x: np.array, bb, pc=0.25, sea_cond=np.array([12, 40, 20, 300]), toh=1e-3, B=4000, eta=1, Rc=0.5,
                            pb=0.1) -> np.array:
            """This method can be used to calculate the gradient for any given Xi."""
            m = x[0]
            N = x[1]
            M = x[2]
            p = 1 + pc
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                + (1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            L = (toh + td) * B
            aa = 50

            dJdN = -1 / N + ((m * p) / (L + N * m * p)) + bb * np.exp((N - 2000) * bb) - bb * np.exp((400 - N) * bb)
            dJdm = -1 / m + ((N * p) / (L + N * m * p)) - bb * np.exp((1 - m) * bb) + bb * np.exp((m - 40) * bb)
            dJdM = -1 / (M * np.log(2) * np.log(M)) + bb * np.exp((M-64)*bb) - bb * np.exp((2 - M)*bb)

            return np.array([dJdm, dJdN, dJdM])

        @staticmethod
        def get_hessian_fn(x: np.array, bb, pc=0.25, *, sea_cond=np.array([12, 40, 20, 300]), toh=1e-3, B=4000, Rc=0.5,
                           eta=1,
                           pb=1) -> np.array:
            """This method can be used to calculate the hessian for any given Xi."""
            m = x[0]
            N = x[1]
            M = x[2]
            td = sea_cond[3] / (1449.2 + 4.6 * sea_cond[0] - 0.055 * (sea_cond[0] ** 2) + 0.00029 * (sea_cond[0] ** 3)
                                + (1.34 - 0.01 * sea_cond[0]) * (sea_cond[1] - 35) + 0.16 * sea_cond[2])
            L = (toh + td) * B
            p = 1 + pc
            aa = 50

            d2JdN2 = -((m ** 2) * (p ** 2)) / (L + N * m * p) ** 2 + 1 / N ** 2 \
                     + (bb ** 2) * np.exp((N - 2000) * bb) + (bb ** 2) * np.exp((400 - N) * bb)
            d2Jdm2 = -((N ** 2) * (p ** 2)) / ((L + N * m * p) ** 2) + 1 / (m ** 2) + (bb**2) * np.exp((1 - m) * bb) + (bb**2) * np.exp((m - 40) * bb)
            d2JdmdN = -((N * m) * (p ** 2)) / (L + N * m * p) ** 2 + p / (L + N * m * p)
            d2JdM2 = 1 / ((M ** 2) * np.log(M) * np.log(2)) + 1 / ((M * np.log2(M) * np.log(
                2)) ** 2) + (bb ** 2) * np.exp((M - 64) * bb) + (bb ** 2) * np.exp((2 - M) * bb)

            return np.array([[d2Jdm2, d2JdmdN, 0],
                             [d2JdmdN, d2JdN2, 0],
                             [0,    0,  d2JdM2]])
            '''return np.array([[d2Jdm2, d2JdmdN],
                             [d2JdmdN, d2JdN2]])'''



