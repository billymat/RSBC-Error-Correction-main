from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os
import sys




def stab_rounds(state, n_rounds, corrector1, corrector2):
    for i in range(n_rounds):
        state = tensor(ket2dm(basis(2, 0)), state)
        psi_new = (corrector2 * corrector1) * state * (corrector2 * corrector1).dag()
        psi_new = psi_new / psi_new.tr()
        psi_new = psi_new.ptrace(1)
        state = psi_new

    return state

