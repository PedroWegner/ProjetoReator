import numpy as np
from scipy.integrate import solve_ivp
from peng_robinson_eos import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from copy import deepcopy

def viscosity_pures(T, C):
    C = np.asarray(C)
    mu = C[:,0] * T ** C[:,1] / (1 + C[:,2] / T + C[:,3] / T**2)
    return np.array(mu)

def viscosity_mixture(T, M, y, C):
    mu = np.asarray(viscosity_pures(T, C))
    M = np.asarray(M)
    y = np.asarray(y)

    # Calcula o phi_ij
    mu_i, mu_j = np.meshgrid(mu, mu)
    M_i, M_j = np.meshgrid(M, M)
    epsilon = 1e-15
    M_i = np.maximum(M_i, epsilon)
    mu_j = np.maximum(mu_j, epsilon)
    M_j = np.maximum(M_j, epsilon)
    numerador = (1 + (mu_i / mu_j)**0.5 * (M_j / M_i)**0.25)**2
    denominador = (8 * (1 + M_i / M_j))**0.5
    phi_ij = numerador / denominador

    # Calcula a viscosidade da misutra
    numerador_terms = y * mu
    denominador_terms = phi_ij @ y
    fraction_terms = numerador_terms / denominador_terms
    mu_mixture = np.sum(fraction_terms)
    return mu_mixture

def calcula_constantes(T, R: float=8.314):
    # Calcula as constantes cinéticas e de adsorção
    k1 = 7200 # kmol kg-1 h-1
    k2 = 360 # kmol kg-1 h-1
    kE = 0.97 * np.exp(- 7261 / (R * T)) # Pa-1
    kO = 3.85 * np.exp(- 96787 / (R * T)) # Pa-1
    return [k1, k2, kE, kO]

def calculate_rate(P_E: float, P_O: float, T: float, R: float=8.314) -> np.ndarray:
    k1, k2, kE, kO = calcula_constantes(T=T, R=R)
    r_E_1 = k1 * kE * kO * P_E * P_O / (kE * P_E + kO * P_O)
    r_E_2 = k2 * kE * kO * P_E * P_O / (kE * P_E + kO * P_O)
    return r_E_1, r_E_2

def calculate_F_system(r_E_1: float, r_E_2: float) -> np.ndarray:
    dFE = - (r_E_1 + r_E_2)
    dFO = - ((1/2) * r_E_1 + 3 * r_E_2)
    dFN = 0
    dFETAL = r_E_1
    dFW = r_E_1 + 3 * r_E_2
    dFDC = 2 * r_E_2
    dF = [dFE, dFO, dFN, dFETAL, dFW, dFDC]
    return np.array(dF)

def calculate_beta_0(G: float, epsilon: float, rho_0: float, dp: float, mu: float) -> float:
    beta_1 = G * (1 - epsilon) / (rho_0 * dp * epsilon**3)
    beta_2 = (150 * (1 - epsilon) * mu) / dp + 1.75 * G
    beta_0 = beta_1 * beta_2
    return beta_0 # Pa-1

def calculate_alpha(beta_0: float, A_col: float, rho_cat: float, epislon: float, P_0: float) -> float:
    alpha = 2 * beta_0 / (A_col * rho_cat * (1 - epislon) * P_0)
    return alpha

def pressure_edo(alpha: float, P: float, P_0: float, F: float, F_0_t: float) -> float:
    dP = - (alpha / 2) * (P_0 / (P / P_0)) * (F / F_0_t)  
    return dP



def edo_system(W: float, 
               vars: np.ndarray, 
               T: float, 
               P_0: float, 
               M_i: np.ndarray,
               F_0: np.ndarray, 
               m_dot: float, 
               rho_0: float, 
               rho_cat: float,
               dp: float, 
               epislon: float, 
               C: np.ndarray,
               D_col: float,
               R: float = 8.314):
    # desempacota as variaveis
    F_E, F_O, F_N, F_ETAL, F_W, F_DC, P = vars
    F = np.asarray([F_E, F_O, F_N, F_ETAL, F_W, F_DC])
    # Desempacota vazoes e calcula composicao
    F_t = np.sum(F)
    y = F / F_t
    P_E, P_O, _, _, _, _= P * y
    # Calcula a taxa cinetica
    r_E_1, r_E_2 = calculate_rate(P_E=P_E, P_O=P_O, T=T, R=R)
    
    # Calcula o sistema de EDO de vazao molar
    dF = calculate_F_system(r_E_1=r_E_1, r_E_2=r_E_2)
     
    # Calcula viscosidade
    mu = viscosity_mixture(T=T, M=M_i, y=y, C=C)
    # Calcula a EDO da pressao
    A_col = D_col**2 * np.pi / 4
    G = (m_dot / A_col) / 3600 # kg m-2 s-1


    beta_0 = calculate_beta_0(G=G, epsilon=epislon, rho_0=rho_0, dp=dp, mu=mu)
    alpha = calculate_alpha(beta_0=beta_0, A_col=A_col, rho_cat=rho_cat, epislon=epislon, P_0=P_0)
    dP = pressure_edo(alpha=alpha, P=P, P_0=P_0, F=F_t, F_0_t=np.sum(F_0))

    EDO = np.append(dF, dP)
    return np.array(EDO)


def FO(W, args):
    W_span = [0.0, W[0]]
    params, X_E_spec = args
    P_0 = params[1]
    F_0 = params[3]
    y0 = np.append(F_0, P_0) # aqui tem que entrar com a pressao e vazao inicial
    sol = solve_ivp(fun=edo_system,
                t_span=W_span,
                y0=y0,
                args=params,
                dense_output=True,
                rtol=1e-8,
                atol=1e-10)
    F_E_final = sol.y[0][-1]
    X_E_calc = (F_0[0] - F_E_final) / F_0[0]
    return 10**4 * (X_E_calc - X_E_spec)**2

def varies_diameters(D, params_ref):
    params = deepcopy(params_ref)
    params += (D, )
    # (T, P_0, M_i, F_0, m_dot, rho_0, rho_cat, dp, epsilon, C, D)
    X_E_spec = 0.79
    args = (params, X_E_spec)
    W_min = minimize(fun=FO, x0=[850], args=(args,), method='NELDER-MEAD')

    # Para pegar a pressao
    W_optmized = W_min.x[0]
    W_span = np.array([0.0, W_optmized])
    P_0 = params[1]
    F_0 = params[3]
    y0 = np.append(F_0, P_0)
    sol = solve_ivp(fun=edo_system,
                    t_span=W_span,
                    y0=y0,
                    args=params,
                    dense_output=True,
                    rtol=1e-8,
                    atol=1e-10)

    W_plot = np.linspace(0.0, W_optmized, 150)
    P_plot = sol.sol(W_plot)
    _, _, _, _, _, _, P = P_plot
    delta_P = P_0 - P[-1]
    print(W_optmized, delta_P, D)
    return W_optmized, delta_P / 101325

if __name__ == '__main__':
    # Instanciando uma engine de Peng-Robinson
    PengRobinsonEngine = ModeloPengRobinson()

    # Defini os parametros para calculo de viscosidade
    C_E = [1.0613E-07, 0.8066, 52.7, 0.0]
    C_O = [1.101e-6, 0.5634, 96.3, 0.0]
    C_N = [6.5592E-07, 0.6081, 54.714, 0.0]
    C_ETAL = [1.9703e-5, 0.17649, 1564.6, 0.0]
    C_W = [1.7096E-08, 1.1146, 0.0, 0.0]
    C_DC = [2.148e-6, 0.46, 290.0, 0.0]
    C = [C_E, C_O, C_N, C_ETAL, C_W, C_DC]
    
    # Definindo os componentes
    ethanol = Component(name='Ethanol', Tc=513.9, Pc=61.48e5, omega=0.645, M=46.069)
    oxygen = Component(name='Oxygen', Tc=154.6, Pc=50.43e5, omega=0.022, M=31.999)
    nitrogen = Component(name='Nitrogen', Tc=126.2, Pc=34.00e5, omega=0.038, M=28.014)
    acetaldehyde = Component(name='Acetaldehyde', Tc=466.0, Pc=55.5e5, omega=0.291, M=44.053)
    water = Component(name="Water", Tc=647.1, Pc=220.55e5, omega=0.345, M=18.015)
    carbon_dioxide = Component(name='Carbon Dioxide', Tc=304.2, Pc=73.83e5, omega=0.224, M=44.01)

    # Parametro binario para Peng-Robinson
    k_Ej = [0, 0.434757, 0.35788, 0, -0.05, 0.095]
    k_Oj = [0.434757, 0, -0.012, 0, 0.355, 0]
    k_Nj = [0.357488, -0.012, 0, 0.1, 0.38, -0.02]
    k_ETALj = [0, 0, 0.1, 0, 0, 0]
    k_Wj = [-0.05, 0.355, 0.38, 0, 0, -0.29873]
    k_DCj = [0.095, 0, -0.02, 0, -0.29873, 0]
    k_ij = np.array([k_Ej, k_Oj, k_Nj, k_ETALj, k_Wj, k_DCj])

    # Instanciando mistura padrao
    mixture = Mixture([ethanol, oxygen, nitrogen, acetaldehyde, water, carbon_dioxide], k_ij=k_ij, l_ij=0.0)
    
    # Parametros da alimentacao
    T = 513 # K
    P_0 = 101325 * 4.6 # Pa
    Far_0 = 1081 # kmol h-1
    FO_0 = 0.21 * Far_0 # kmol h-1
    FN_0 = 0.79 * Far_0 # kmol h-1
    FE_0 = 227 # kmol h-1
    F_0 = np.array([FE_0, FO_0, FN_0, 0.0, 0.0, 0.0]) # kmol h-1
    y_0 = F_0 / np.sum(F_0) # adimensional
    # instanciando um estado padrao
    state = State(mixture=mixture, T=T, P=P_0, z=y_0, is_vapor=True)
    M_i = [c.M for c in mixture.components]
    M_mix_0 = np.sum(y_0 * M_i) # kg / kmol
    m_dot = np.sum(F_0) * M_mix_0   # kg h-1
    Z_0 = max(PengRobinsonEngine._get_Z(state=state)) # adimensional
    rho_0 = P_0 * (M_mix_0 / 1000) / (8.314 * Z_0 * T) # kg m-3
    Q_0_dot = m_dot / rho_0 # m3 h-1
    
    # Parametros do catalisador e do leito
    dp = 6.35e-3 # m
    epsilon = 0.815 # adimensional
    rho_cat = 6054 # kg m-3
    rho_buld = 1120 # kg m-3
    D_col = 0.848274624 # m

    # Parametros basicos
    params = (T, P_0, M_i, F_0, m_dot, rho_0, rho_cat, dp, epsilon, C) # talvez tenha que incluir o estado aqui
    
    X_E_spec = 0.79



    