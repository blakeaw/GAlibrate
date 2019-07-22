'''
Dimerization model:
 A + A <> AA
'''
from pysb import Model, Parameter, Monomer, Rule, Observable, Initial


Model()
#######
V = 100.
#######
Parameter('kf',   0.001)
Parameter('kr',   1.)


Monomer('A', ['d'])

# Rules
Rule('ReversibleBinding', A(d=None) + A(d=None) | A(d=1) % A(d=1), kf, kr)

#Observables
Observable("A_free", A(d=None))
Observable("A_dimer", A(d=1) % A(d=1))

# Inital Conditions
Parameter("A_0", 20.*V)
Initial(A(d=None), A_0)
