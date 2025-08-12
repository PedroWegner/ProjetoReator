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
    params, F_0, X_E_specificada = args
    sol = solve_ivp(fun=edo_system,
                t_span=W_span,
                y0=F_0,
                args=params,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10)
    F_E_final = sol.y[0][-1]
    X_E_calc = (F_0[0] - F_E_final) / F_0[0]
    return (X_E_calc - X_E_specificada)**2


if __name__ == '__main__':
    Far_0 = 1081
    FO_0 = 0.21 * Far_0
    FN_0 = 0.79 * Far_0
    FE_0 = 227
    F_0 = np.array([FE_0, FO_0, FN_0, 0.0, 0.0, 0.0])
    T = 513
    P = 101325 * 4.6
    params = (T, P,)


    # Variaveis para otimizacao
    X_E_spec = 0.79
    args = (params, F_0, X_E_spec,)

    W_minimization = minimize(fun=FO, x0=[700], args=(args,), method='NELDER-MEAD')
    W_optmized = W_minimization.x[0]
    W_span = np.array([0.0, W_optmized])
    print(W_optmized)
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
    F_E_plot, F_O_plot, F_N_plot, F_ETAL_plot, F_W_plot, F_DC_plot = F_plot
    F_t_plot = np.sum(F_plot[:-1], axis=0)
    y_E_plot, y_O_plot, y_N_plot, y_ETAL_plot, y_W_plot, y_DC_plot = F_plot / F_t_plot
    print(F_plot[:,-1])
    plt.plot(W_plot, y_E_plot, label=r'$y_{E}$')
    plt.plot(W_plot, y_O_plot, label=r'$y_{O}$')
    plt.plot(W_plot, y_ETAL_plot, label=r'$y_{ETAL}$')
    plt.plot(W_plot, y_W_plot, label=r'$y_{W}$')
    plt.plot(W_plot, y_DC_plot, label=r'$y_{DC}$')
    plt.plot(W_plot, y_N_plot, label=r'$y_{N}$')
    plt.xlabel(r'$W_{cat}\;[kg]$')
    plt.xlim(left=0.0, right=W_optmized)
    plt.ylim(bottom=0.0)
    plt.ylabel(r'$y_{i}$') #\;[kmol\;h^{-1}]
    plt.legend()
    plt.show()
