from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

RGAS_SI = 8.314 # constantes dos gases J mol-1 K-1
@dataclass
class Component:
    name: str
    Tc: float
    Pc: float
    omega: float
    M: float

@dataclass
class Mixture:
    components: list[Component]
    k_ij: np.ndarray
    l_ij: np.ndarray

@dataclass
class State:
    mixture: Mixture
    z: np.ndarray
    is_vapor: bool
    T: float
    P: Optional[float] = None
    Z: Optional[float] = None
    Vm: Optional[float] = None
    V: Optional[float] = None
    n: float = 100
    helmholtz_derivatives: Optional[Dict[str, any]] = None
    P_derivatives: Optional[Dict[str, any]] = None
    fugacity_dict: Optional[Dict[str, any]] = None
    residual_props: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, any]] = None
    
class CubicParametersWorker:
    def __init__(self, omega1: float, omega2: float, m):
        self.omega1 = omega1
        self.omega2 = omega2
        self.m = m

        self.mixture = None
        self.components = None
        self.z = None
        self.n = None
        self.Tc = None
        self.Pc = None
        self.omega = None
        self.T = None
        self.Tr = None
        self.params_dict = {}

    def _calculate_pure_params(self) -> None:
        m = self.m(self.omega)
        alpha = (1 + m * (1 - np.sqrt(self.Tr)))**2
        ac = self.omega1 * (RGAS_SI * self.Tc)**2 / self.Pc
        ai = ac * alpha
        bi = self.omega2 * (RGAS_SI * self.Tc) / self.Pc 
        
        self.params_dict['m']= m
        self.params_dict['alpha']= alpha
        self.params_dict['ac']= ac
        self.params_dict['ai']= ai
        self.params_dict['bi']= bi

    def _calculate_binary_mixture_params(self):
        ai = self.params_dict['ai']
        bi = self.params_dict['bi']
        aij_matrix = (np.sqrt(np.outer(ai, ai))) * (1 - self.mixture.k_ij)
        bij_matrix = 0.5 * (np.add.outer(bi,bi)) * (1 - self.mixture.l_ij)
        a_mix = self.z @ aij_matrix @ self.z
        b_mix = self.z @ bij_matrix @ self.z

        self.params_dict['aij_matrix'] =  aij_matrix
        self.params_dict['bij_matrix'] =  bij_matrix
        self.params_dict['a_mix'] =  a_mix
        self.params_dict['b_mix'] =  b_mix

    def _calculate_B_and_derivatives(self):
        ni = self.z * self.n
        bij_matrix = self.params_dict['bij_matrix']
        b_mix = self.params_dict['b_mix']
        B = self.n * b_mix
        Bi = np.array((2 * bij_matrix @ ni - B) / self.n)
        soma_BiBj = Bi.reshape(-1, 1) + Bi.reshape(1, -1)
        Bij = (2 * bij_matrix - soma_BiBj) / self.n

        self.params_dict['B'] =  B
        self.params_dict['Bi'] =  Bi
        self.params_dict['Bij'] =  Bij
        
    def _calculate_D_and_derivatives(self):
        ni = np.array(self.z * self.n)
        ai = self.params_dict['ai']
        aij_matrix = self.params_dict['aij_matrix']
        alpha = self.params_dict['alpha']
        m = self.params_dict['m']
        ac = self.params_dict['ac']
        a_mix = self.params_dict['a_mix']
        D = self.n**2 * a_mix
        Di = 2 * (ni @ aij_matrix)
        Dij = 2 * aij_matrix

        alphaij_T = ac * (- m * (alpha * self.Tr)**0.5) / self.T
        aii_ajj = np.outer(ai, ai)
        aii_dajj = np.outer(ai, alphaij_T)
        ajj_daii = np.outer(alphaij_T, ai)
        aij_T = (1 - self.mixture.k_ij) *(aii_dajj + ajj_daii) / (2 * aii_ajj**0.5)

        DiT = 2 * ni @ aij_T
        DT = (1/2) * ni @ DiT

        alphaii_TT = ac * m * (1 + m) * self.Tr**0.5 / (2 * self.T**2)
        # Eq. 105
        delh_delT = - (1 / (2 * (aii_ajj)**(3 / 2))) * (aii_dajj + ajj_daii)**2
        daii_dajj = np.outer(alphaij_T, alphaij_T)
        aii_ddajj = np.outer(ai, alphaii_TT)
        ajj_ddaii = np.outer(alphaii_TT, ai)
        delg_delT = (2 * daii_dajj + aii_ddajj + ajj_ddaii) * (1 / aii_ajj**0.5)
        daij_TT = ((1 - self.mixture.k_ij) / 2) * (delh_delT + delg_delT)
        DTT = ni @ daij_TT @ ni

        self.params_dict['D'] =  D
        self.params_dict['Di'] =  Di
        self.params_dict['DiT'] =  DiT
        self.params_dict['Dij'] =  Dij
        self.params_dict['DT'] =  DT
        self.params_dict['DTT'] =  DTT

    def _allocate_variables(self, state: State) -> None:
        self.mixture = state.mixture
        self.components = state.mixture.components
        self.z = state.z
        self.n = state.n
        self.Tc = np.array([c.Tc for c in self.components])
        self.Pc = np.array([c.Pc for c in self.components])
        self.omega = np.array([c.omega for c in self.components])
        self.T = state.T 
        self.Tr = self.T / self.Tc

    def _deallocate_variables(self) -> None:
        self.mixture = None
        self.components = None
        self.z = None
        self.n = None
        self.Tc = None
        self.Pc = None
        self.omega = None
        self.T = None
        self.Tr = None
        self.params_dict = {}

    def params_to_dict(self, state: State):
        # Alloca variaveis necessaria para os calculos
        self._deallocate_variables()
        self._allocate_variables(state=state)
        
        self._calculate_pure_params()
        self._calculate_binary_mixture_params()
        self._calculate_B_and_derivatives()
        self._calculate_D_and_derivatives()
        """Empacota todos os calculos do worker para enviar para o strategy"""
        return self.params_dict

class SolveZWorker:
    def __init__(self, delta1: float, delta2: float):
        self.delta1 = delta1
        self.delta2 = delta2
        
    def _solver_Z(self, A: float, B: float):
        # 1. Parametros do modelos
        delta = self.delta1 + self.delta2
        delta_inv = self.delta1 * self.delta2
        # 2. Solucao analitica da cubica
        a1 = B * (delta - 1) - 1
        a2 = B**2 * (delta_inv - delta) - B * delta + A
        a3 = - (B**2 * delta_inv * (B + 1) + A * B)
        _Q = (3 * a2 - a1**2) / 9
        _R = (9 * a1 * a2 - 27 * a3 -2 *a1**3)/54
        _D = _Q**3 + _R**2
        if _D < 0:
            theta = np.arccos(_R / np.sqrt(-_Q**3))
            x1 = 2 * np.sqrt(-_Q) * np.cos(theta / 3)  - a1 /3
            x2 = 2 * np.sqrt(-_Q) * np.cos((theta + 2 * np.pi) / 3) - a1 /3
            x3 = 2 * np.sqrt(-_Q) * np.cos((theta + 4 * np.pi) / 3) - a1 /3
        else:
            _S = np.cbrt(_R + np.sqrt(_D))
            _T = np.cbrt(_R - np.sqrt(_D))
            x1 = _S + _T - (1/3) * a1
            x2 = (-1/2)*(_S + _T) - (1/3) * a1 + (1/2) * 1j * np.sqrt(3) * (_S - _T)
            x3 = (-1/2)*(_S + _T) - (1/3) * a1 - (1/2) * 1j * np.sqrt(3) * (_S - _T)
        # 3. Limpeza das raizes obtidas
        Z = [x1, x2, x3]
        print(Z)
        Z = [r.real for r in Z if np.isclose(r.imag, 0) and r.real > 0]
        return sorted(Z)

    def get_Z_(self, state: State) -> tuple:
        A = state.params['a_mix'] * state.P / (RGAS_SI * state.T)**2
        B = state.params['b_mix'] * state.P / (RGAS_SI * state.T)
        Z = self._solver_Z(A=A, B=B)
        return Z

    def get_Z(self, state: State, params) -> tuple:
        state = state
        # print(state.params)
        A = params['a_mix'] * state.P / (RGAS_SI * state.T)**2
        B = params['b_mix'] * state.P / (RGAS_SI * state.T)
        Z = self._solver_Z(A=A, B=B)
        is_vapor = None
        if len(Z) == 1:
            # print("O sistema só tem uma fase possível")
            if Z[0] < 0.5:
                # print("O estado só pode ser liquido")
                is_vapor = False
            else: 
                # print("O sistema só pode ser vapor")
                is_vapor = True
        else: 
            is_vapor = state.is_vapor
        return (Z, is_vapor)

class ModeloPengRobinson: #essa classe pode ser quebrada para adotar outras EoS cubicas!!!!
    def __init__(self):
        # 1. Parametros universais do modelo de Peng-Robinson
        self.delta1 = 1 + np.sqrt(2)
        self.delta2 = 1 - np.sqrt(2)
        self.omega1 = 0.45724
        self.omega2 = 0.07780
        self.m_func = lambda omega: 0.37464 + 1.54226 * omega - 0.26992 * omega**2

        # Inicializando os workers da classe
        self.params_worker = CubicParametersWorker(omega1=self.omega1, omega2=self.omega2, m=self.m_func)
        self.solver_Z_worker = SolveZWorker(delta1=self.delta1, delta2=self.delta2)

    def calculate_params(self, state: State) -> None:
        state.params = self.params_worker.params_to_dict(state=state)

    def _get_Z(self, state: State) -> np.ndarray:
        params = self.params_worker.params_to_dict(state=state)
        state.params = params
        Z = self.solver_Z_worker.get_Z_(state=state) # ponto de apoio
        return Z


        

if __name__ == '__main__':
    # 1. Instancia os componentes teste
    benzeno = Component(name='Benzeno', Tc=562.05, Pc=48.95e5, omega=0.21030)
    tolueno = Component(name='Tolueno', Tc=591.75, Pc=41.08e5, omega=0.26401)
    metano = Component(name='Methane', Tc=190.6, Pc=45.99e5, omega=0.012)    
    dioxide = Component(name='Carbon Dioxide', Tc=304.2, Pc=73.83e5, omega=0.224)
    nitrogenio = Component(name='Nitrogen', Tc=126.00, Pc=34.00e5, omega=0.038)
    # 2. Instancia a mistura
    k_ij = 0.093
    k_ij = np.array([[0, k_ij],[k_ij,0]])
    mixture = Mixture([metano, dioxide], k_ij=k_ij, l_ij=0.0)

    # 3. Condicoes do flash
    T = 200 # K
    P = 30e5 # Pa
    z = np.array([0.5, 0.5])
    trial_state = State(mixture=mixture, T=T, z=z, is_vapor=True)
