from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def stab_rounds(state, n_rounds, corrector1, corrector2):
    for i in range(n_rounds):
        state = tensor(ket2dm(basis(2, 0)), state)
        psi_new = (corrector2 * corrector1) * state * (corrector2 * corrector1).dag()
        psi_new = psi_new / psi_new.tr()
        psi_new = psi_new.ptrace(1)
        state = psi_new

    return state


N = 2
cutoff = 13
n_operator = num(cutoff)
a = destroy(cutoff)
phi_operator = fock(cutoff, 0)*fock(cutoff, N).dag()
ancilla_plus = (basis(2, 0) + basis(2, 1)).unit()
ancilla_minus = (basis(2, 0) - basis(2, 1)).unit()

logical0 = (fock(cutoff, 2*N) + fock(cutoff, 4*N) + fock(cutoff, 6*N)).unit()
logical1 = (fock(cutoff, N) + fock(cutoff, 3*N) + fock(cutoff, 5*N)).unit()

logical_plus = (logical0 + logical1).unit()
logical_minus = (logical0 - logical1).unit()


for i in range(1, cutoff-N):
    phi_operator += fock(cutoff, i) * fock(cutoff, i+N).dag()

S_l = fock(cutoff, 0)*fock(cutoff, 2*N).dag()
for i in range(1, cutoff-2*N):
    S_l += fock(cutoff, i) * fock(cutoff, i+2*N).dag()

coupled_S_l = tensor(ket2dm(ancilla_plus), S_l) + tensor(ket2dm(ancilla_minus), S_l.dag())

ln_Z = 1j * np.pi * n_operator / N
Z_L = ln_Z.expm()


#Simulation

H = 0 * a.dag() * a
delta_t = 1
coupled_Z = tensor(sigmay(),  0.95 * ln_Z).expm()
psi0 = logical0
psi = ket2dm(psi0)
fidelity_array = [1]
time_array = delta_t * np.arange(31)

for x in range(30):
    times = np.linspace(0.0, delta_t, 100)
    result = mesolve(H, psi, times, [np.sqrt(0.018) * a.dag() * a])

    psi = result.states[-1]
    psi = stab_rounds(psi, 6, coupled_S_l, coupled_Z)

    print(expect(ln_Z.expm(), psi))
    print(expect(S_l, psi))
    print(fidelity(psi0, psi))