
from pysb import Model, Parameter, Monomer, Rule, Observable, Initial
from scipy.stats import norm
import numpy as np
from galibrate.pysb import GaoIt, GAlibrateIt



SHARED = dict()

def test_gaoit_init():
    SHARED['gao_it'] = GaoIt()

def test_gaoit_addbycall():
    # We'll build a simple dimeration model
    # with this call:
    #    Dimerization model:
    #        A + A <> AA
    Model()
    #######
    V = 100.
    #######
    SHARED['gao_it'](Parameter('kf',   0.001))
    SHARED['gao_it'](Parameter('kr',   1.), loc=np.log10(1.)-1., width=2.)

    Monomer('A', ['d'])

    # Rules
    Rule('ReversibleBinding', A(d=None) + A(d=None) | A(d=1) % A(d=1), kf, kr)

    #Observables
    Observable("A_free", A(d=None))
    Observable("A_dimer", A(d=1) % A(d=1))

    # Inital Conditions
    Parameter("A_0", 20.*V)
    Initial(A(d=None), A_0)
    assert len(SHARED['gao_it'].parms) == 2

def test_gaoit_isub():

    SHARED['gao_it'] -= model.parameters['kf']
    assert len(SHARED['gao_it'].parms) == 1
    SHARED['gao_it'] -= 'kr'
    assert len(SHARED['gao_it'].parms) == 0

def test_gaoit_func_add_all_kinetic_params():
    SHARED['gao_it'].add_all_kinetic_params(model)
    assert len(SHARED['gao_it'].parms) == 2

if __name__ == '__main__':
    test_gaoit_init()
    test_gaoit_addbycall()
    test_gaoit_isub()
    test_gaoit_func_add_all_kinetic_params()

