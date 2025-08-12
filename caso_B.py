import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def edo(W, X, T, P, theta_O, PE_0, FE_0, epsilon_E: np.ndarray=None, R:float=8.314):
    # Calcula as constantes cinéticas e de adsorção
    k1 = 7200 # kmol kg-1 h-1
    kE = 0.97 * np.exp(- 7261 / (R * T))
    kO = 3.85 * np.exp(- 96787 / (R * T))

    exp_vol = 1 + epsilon_E * X
    PE = PE_0 * (1 - X) / exp_vol
    PO = PE_0 * (theta_O - 0.5 * X) / exp_vol
    r_E = k1 * kE * kO * PE * PO / (kE * PE + kO * PO)
    edo = r_E / FE_0
    return edo


def FO(W, args):
    W_span = [0.0, W[0]]
    X_E = np.array([0.0])
    params, X_E_spec = args
    sol = solve_ivp(fun=edo,
                t_span=W_span,
                y0=X_E,
                args=params,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10)
    return 10**4 * (sol.y[0][-1] - X_E_spec)**2


# Parametros especificados
Far_0 = 1081
FO_0 = 0.21 * Far_0
FN_0 = 0.79 * Far_0
FE_0 = 227
F_0 = np.array([FE_0, FO_0, FN_0, 0.0, 0.0])
Ft_0 = np.sum(F_0)
T = 513
P = 101325 * 4.6
theta_O= FO_0 / FE_0
PE_0= P * FE_0 / Ft_0
delta = 1 + 1 - 0.5 - 1
epsilon_E = delta * FE_0 / Ft_0
params = (T, P, theta_O, PE_0, FE_0, epsilon_E)


# Conversao especificada (esperada)
X_E_spec = 0.79

args = (params, X_E_spec,)


W_minimization = minimize(fun=FO, x0=[500], args=(args,), method='NELDER-MEAD')
W_optmized = W_minimization.x[0]
W_span = np.array([0.0, W_optmized])
print(W_optmized)

sol = solve_ivp(fun=edo,
                t_span=W_span,
                y0=np.array([0.0]),
                args=params,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10)

# dominios para gerar o grafico
print(W_optmized)
W_plot = np.linspace(0.0, W_optmized, 1500)
X_plot = sol.sol(W_plot)[0]
exp_vol = 1 + epsilon_E * X_plot
PE_plot = (PE_0 * (1 - X_plot))/ (exp_vol * 101325)
PO_plot = (PE_0 * (theta_O - 0.5 * X_plot))/ (exp_vol * 101325)
PETAL_plot = (PE_0 * X_plot)/ (exp_vol * 101325)
PW_plot = (PE_0 * X_plot)/ (exp_vol * 101325)
PN_plot = 4.6 - PE_plot - PO_plot - PETAL_plot - PW_plot

plt.plot(W_plot, PE_plot/4.6, label=r'$y_{E}$')
plt.plot(W_plot, PO_plot/4.6, label=r'$y_{O}$')
plt.plot(W_plot, PETAL_plot/4.6, linestyle='dashed', label=r'$y_{ETAL}$')
plt.plot(W_plot, PW_plot/4.6, linestyle='dotted', label=r'$y_{W}$')
plt.plot(W_plot, PN_plot/4.6, label=r'$y_{W}$')
plt.xlabel(r'$W_{cat}\;[kg]$')
plt.xlim(left=0.0, right=W_optmized)
plt.ylim(bottom=0.0)
plt.ylabel(r'$y_{i}$')
plt.legend()
plt.show()
