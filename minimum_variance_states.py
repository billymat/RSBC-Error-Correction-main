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
F = 100
cutoff = F+20

#Operators
n_operator = num(cutoff)
a = destroy(cutoff)
ln_Z = 1j * np.pi * n_operator / N
Z_L = ln_Z.expm()

#Bases
logical_plus = np.sin((N * np.pi)/(F + 2*N - F % N)) * fock(cutoff, 0)

for k in range(1, F//N + 1):
    logical_plus += np.sin((k*N + N) * np.pi/(F + 2*N - F % N)) * fock(cutoff, k*N)

logical_plus = logical_plus.unit()

logical_minus = np.sin((N * np.pi)/(F + 2*N - F % N)) * fock(cutoff, 0)

for k in range(1, F//N + 1):
    logical_minus += (-1)**k * np.sin((k*N + N) * np.pi/(F + 2*N - F % N)) * fock(cutoff, k*N)

logical_minus = logical_minus.unit()


logical0 = (logical_plus + logical_minus).unit()
logical1 = (logical_plus - logical_minus).unit()

ancilla_plus = (basis(2, 0) + basis(2, 1)).unit()
ancilla_minus = (basis(2, 0) - basis(2, 1)).unit()

#Ladder operators
phi_operator = fock(cutoff, 0)*fock(cutoff, N).dag()
for i in range(1, cutoff-N):
    phi_operator += fock(cutoff, i) * fock(cutoff, i+N).dag()

S_l = fock(cutoff, 0)*fock(cutoff, 2*N).dag()
for i in range(1, cutoff-2*N):
    S_l += fock(cutoff, i) * fock(cutoff, i+2*N).dag()


#Just checking
print(logical0.overlap(logical1))
print(expect(n_operator, logical_plus))
print(expect(S_l, logical_plus))
print(expect(phi_operator, logical_plus))

#Correction?
coupled_S_l = tensor(ket2dm(ancilla_plus), S_l) + tensor(ket2dm(ancilla_minus), S_l.dag())

coupled_Z = tensor(sigmay(), 0.985*ln_Z).expm()

'''psi0 = logical0
psi = ket2dm(psi0)
range_epsilon = np.linspace(-1, 1, 400)
for x in range_epsilon:
    coupled_Z = tensor(sigmay(), x * ln_Z).expm()
    psi = stab_rounds(psi, 1, coupled_S_l, coupled_Z)
    print(x, fidelity(psi, logical0))'''


H = 0 * a.dag() * a
delta_t = 0.1
psi0 = logical0
psi = ket2dm(psi0)
fidelity_array = [1]
S_l_array = [expect(S_l, psi0).real]
time_array = delta_t * np.arange(31)

for x in range(1, 401):
    times = np.linspace(0.0, delta_t, 200)
    result = mesolve(H, psi, times, [np.sqrt(0.001) * a, np.sqrt(0.018) * a.dag() * a])

    psi = result.states[-1]

    if x % 2 == 0:
        alpha = np.random.normal(0, 0.01) + 1.j * np.random.uniform(0, 0.01)
        D = displace(cutoff, alpha)
        psi = D * psi * D.dag()

    if x % 10 == 0:
        psi = stab_rounds(psi, 6, coupled_S_l, coupled_Z)

    S_l_array.append(expect(S_l, psi).real)
    fidelity_array.append(fidelity(psi0, psi))

print(S_l_array)
print(fidelity_array)


psi0 = logical0
psi = ket2dm(psi0)
fidelity_array_2 = [1]
S_l_array_2 = [expect(S_l, psi0).real]
time_array = delta_t * np.arange(31)

for x in range(400):
    times = np.linspace(0.0, delta_t, 200)
    result = mesolve(H, psi, times, [np.sqrt(0.001) * a, np.sqrt(0.018) * a.dag() * a])

    psi = result.states[-1]
    #psi = stab_rounds(psi, 1, coupled_S_l, coupled_Z)
    if x % 2 == 0:
        alpha = np.random.normal(0, 0.01) + 1.j * np.random.uniform(0, 0.01)
        D = displace(cutoff, alpha)
        psi = D * psi * D.dag()

    S_l_array_2.append(expect(S_l, psi).real)
    fidelity_array_2.append(fidelity(psi0, psi))

print(S_l_array_2)
print(fidelity_array_2)

plt.plot(S_l_array)
plt.plot(S_l_array_2)
plt.plot(fidelity_array)
plt.plot(fidelity_array_2)
plt.show()
