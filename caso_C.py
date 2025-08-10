import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def edo_system(W, F, T, P, R: float = 8.314):
    # Calcula as constantes cinéticas e de adsorção
    k1 = 7200 # kmol kg-1 h-1
    k2 = 360 # kmol kg-1 h-1
    kE = 0.97 * np.exp(- 7261 / (R * T)) # Pa-1
    kO = 3.85 * np.exp(- 96787 / (R * T)) # Pa-1

    # Desempacota vazoes e calcula composicao
    F_t = np.sum(F)
    y = F / F_t
    P_E, P_O, _, _, _, _= P * y

    # Calcula a taxa cinetica
    r_E_1 = k1 * kE * kO * P_E * P_O / (kE * P_E + kO * P_O)
    r_E_2 = k2 * kE * kO * P_E * P_O / (kE * P_E + kO * P_O)

    # Calcula o sistema de EDO
    dFE = - (r_E_1 + r_E_2)
    dFO = - ((1/2) * r_E_1 + 3 * r_E_2)
    dFN = 0
    dFETAL = r_E_1
    dFW = r_E_1 + 3 * r_E_2
    dFDC = 2 * r_E_2
    
    EDO = [dFE, dFO, dFN, dFETAL, dFW, dFDC]
    return np.array(EDO)

def FO(W, args):
    W_span = [0.0, W[0]]
    params, F_0, F_ETAL_spec = args
    sol = solve_ivp(fun=edo_system,
                t_span=W_span,
                y0=F_0,
                args=params,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10)
    F_ETAL_calc = sol.y[3][-1]
    return (F_ETAL_calc - F_ETAL_spec)**2


if __name__ == '__main__':
    Far_0 = 1081
    FO_0 = 0.21 * Far_0
    FN_0 = 0.79 * Far_0
    FE_0 = 227
    F_0 = np.array([FE_0, FO_0, FN_0, 0.0, 0.0, 0.0])
    T = 513
    P = 101325 * 3.6
    params = (T, P,)


    # Variaveis para otimizacao
    F_ETAL_spec = 208.84
    args = (params, F_0, F_ETAL_spec,)

    W_minimization = minimize(fun=FO, x0=[500], args=(args,), method='NELDER-MEAD')
    W_optmized = W_minimization.x[0]
    W_span = np.array([0.0, W_optmized])
    
    sol = solve_ivp(fun=edo_system,
                    t_span=W_span,
                    y0=F_0,
                    args=params,
                    dense_output=True,
                    rtol=1e-8,
                    atol=1e-10)

    # Plotagem do grafico
    W_plot = np.linspace(0.0, W_optmized, 1500)
    F_plot = sol.sol(W_plot)
    F_E_plot, F_O_plot, _, F_ETAL_plot, F_W_plot, F_DC_plot = F_plot

    plt.plot(W_plot, F_E_plot, label=r'$F_{E}$')
    plt.plot(W_plot, F_O_plot, label=r'$F_{O}$')
    plt.plot(W_plot, F_ETAL_plot, linestyle='dashed', label=r'$F_{ETAL}$')
    plt.plot(W_plot, F_W_plot, linestyle='dotted', label=r'$F_{W}$')
    plt.plot(W_plot, F_DC_plot, linestyle='dotted', label=r'$F_{DC}$')
    plt.xlabel(r'$W_{cat}\;[kg]$')
    plt.xlim(left=0.0, right=W_optmized)
    plt.ylim(bottom=0.0)
    plt.ylabel(r'$F_{i}\;[kmol\;h^{-1}]$')
    plt.legend()
    plt.show()
