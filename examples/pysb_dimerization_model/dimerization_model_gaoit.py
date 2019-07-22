'''
Dimerization model:
 A + A <> AA
'''
from pysb import Model, Parameter, Monomer, Rule, Observable, Initial
from scipy.stats import norm
import numpy as np
from galibrate.pysb_utils import GaoIt

gao_it = GaoIt()

Model()
#######
V = 100.
#######
gao_it(Parameter('kf',   0.001))
gao_it(Parameter('kr',   1.), loc=np.log10(1.)-1., width=2.)


Monomer('A', ['d'])

# Rules
Rule('ReversibleBinding', A(d=None) + A(d=None) | A(d=1) % A(d=1), kf, kr)

#Observables
Observable("A_free", A(d=None))
Observable("A_dimer", A(d=1) % A(d=1))

# Inital Conditions
Parameter("A_0", 20.*V)
Initial(A(d=None), A_0)
