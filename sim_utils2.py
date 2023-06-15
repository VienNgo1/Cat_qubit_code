import numpy as np
import qutip as qt

MHz = 1. # work in units of MHz
pi = np.pi

class CQ:
    
    def __init__(self,
                 N = 25,                                      # Dimensionality of the Hilbert space, per resonator
                 nqubits = 1,                                 # Number of cat qubits (i.e. resonators)
                 e_2 = 2. * pi * 360 * 0.01 * MHz,            # Squeezing drive strength
                 tau = 0.32 * MHz,                            # Squeezing drive ramp time
                 e_x_rabi = 2. * pi * 0.74 * MHz,              # Rabi oscillation drive strength
                 T_1 = 1 / (2. * pi * 53. * 0.001 * MHz),     # Single - photon decay time
                 n_th = 0.04):                                # Population of n=1 Fock state in initial thermal state
                                                    
        
        self.N = N
        self.nqubits = nqubits
        self.e_2 = e_2
        self.tau = tau
        self.e_x_rabi = e_x_rabi
        self.T_1 = T_1
        self.n_th = n_th
        self.kappa = 1. / T_1
        
        self.t0 = 0.
        self.alpha = np.sqrt((self.e_2) / self.kappa)
        
        self.beta = 0.2 * np.exp(1j * pi / 4)
        
        self.C_0 = (qt.coherent(self.N, self.alpha) + qt.coherent(self.N, -self.alpha)) / np.sqrt(2. * (1. + np.exp(-2. * np.abs(self.alpha) ** 2)))
        self.C_1 = (qt.coherent(self.N, self.alpha) - qt.coherent(self.N, -self.alpha)) / np.sqrt(2. * (1. - np.exp(-2. * np.abs(self.alpha) ** 2)))
        
        mag0 = np.real((self.C_0.dag() * self.C_0).full().item())
        mag1 = np.real((self.C_1.dag() * self.C_1).full().item())
        
        #Check the normalization to get an idea of whether or not we have a high enough dimensionality
        if mag0 < 0.995 or mag1 < 0.995:
            print("Warning: Hilbert space dimensionality is likely too small for the given system parameters, consider increasing it")
        
        self.C_0 /= np.sqrt(mag0)
        self.C_1 /= np.sqrt(mag1)
        
        self.H = []
        for i in range(self.nqubits):
            self.two_photon(i)
            
        self.t0 += 1.1 * self.tau
        
    def two_photon(self, qubit):
        H_list = [qt.identity(self.N)] * self.nqubits
        H_list[qubit] = self.e_2 * qt.create(self.N) ** 2 + np.conj(self.e_2) * qt.destroy(self.N) **2
        const = 0.5 * np.tanh(-2.) + 0.5
        s = "(0.5 * tanh(4. * (t - {0}) / {1} - 2.) + 0.5 - {2}) / (1. - {2})".format(self.t0, self.tau, const)
        self.H.append([qt.tensor(H_list), s])    
    
#     def loss_terms(self):
#         ret = []
#         for i in range(self.nqubits):
#             H_list_destroy = [qt.identity(self.N)] * self.nqubits 
#             H_list_destroy[i] = qt.destroy(self.N) ** 2 
#             ret.append(np.sqrt(self.kappa * (1. + self.n_th)) * qt.tensor(H_list_destroy))
#         return ret

    def loss_terms(self):
        ret = []
        for i in range(self.nqubits):
            H_list_destroy = [qt.identity(self.N)] * self.nqubits 
            H_list_destroy[i] = qt.destroy(self.N) ** 2 
            ret.append(np.sqrt(self.kappa) * qt.tensor(H_list_destroy))
        return ret
         
    def cat_states(self):
        return [self.C_0, self.C_1]
    
#     def thermal_state(self):
#         rho_single = (1. - self.n_th) * qt.basis(self.N, 0) * qt.basis(self.N, 0).dag() + self.n_th * qt.basis(self.N, 1) * qt.basis(self.N, 1).dag()
#         return qt.tensor([rho_single] * self.nqubits)
    
    def thermal_state(self):
        rho_single = qt.coherent(self.N, self.beta) * qt.coherent(self.N, self.beta).dag() 
        return qt.tensor([rho_single] * self.nqubits)    

    def add_rabi_oscillation(self, qubit):
        H_list = [qt.identity(self.N)] * self.nqubits
        H_list[qubit] = self.e_x_rabi * qt.create(self.N) + np.conj(self.e_x_rabi) * qt.destroy(self.N)
        self.H.append([qt.tensor(H_list), self.pulse(self.t0, 100.)])
       
    def pulse(self, t0, duration, tau=None):
        if tau is None:
            tau =1.0e-5 * duration
        return "0.5 * tanh((t - {0}) / {1}) - 0.5 * tanh((t - {0} - {2}) / {1})".format(t0, tau, duration)
    
    
    
