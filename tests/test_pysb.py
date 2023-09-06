
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

def test_gaoit_getitem():
    loc = np.log10(model.parameters['kf'].value)-2.
    width = 4.
    loc_width = SHARED['gao_it']['kf']
    assert np.allclose((loc, width), loc_width)

def test_gaoit_iadd():
    gao_it = GaoIt()
    gao_it += model.parameters['kf']
    assert len(gao_it.parms) == 1
    assert 'kf' in list(gao_it.parms.keys())    

def test_gaoit_contains():
    assert 'kf' in SHARED['gao_it']
    assert 'kr' in SHARED['gao_it']

def test_gaoit_setitem():
    loc = np.log10(model.parameters['kf'].value)-1.
    width = 2.
    loc_width_old = SHARED['gao_it']['kf']
    SHARED['gao_it']['kf'] = (loc, width)
    loc_width_new = SHARED['gao_it']['kf']
    assert np.allclose(loc_width_new, [loc, width])
    assert (not np.allclose(loc_width_old, loc_width_new))

def test_gaoit_names():
    assert SHARED['gao_it'].names() == ['kf', 'kr']

def test_gaoit_keys():
    assert list(SHARED['gao_it'].keys()) == ['kf', 'kr']

def test_gaoit_mask():
    assert SHARED['gao_it'].mask(model.parameters) == [True, True, False]

def test_gaoit_locs():
    assert len(SHARED['gao_it'].locs()) == 2

def test_gaoit_widths():
    assert len(SHARED['gao_it'].widths()) == 2

def test_gaoit_sampled_parameters():
    assert len(SHARED['gao_it'].sampled_parameters()) == 2

def test_gaoit_add_all_nonkinetic_params():
    gao_it = GaoIt()
    gao_it.add_all_nonkinetic_params(model)
    assert gao_it.mask(model.parameters) == [False, False, True]   

def test_gaoit_add_by_name_with_string_input():
    gao_it = GaoIt()
    # Test adding by name with single string
    gao_it.add_by_name(model, 'kf')
    assert gao_it.mask(model.parameters) == [True, False, False]

def test_gaoit_add_by_name_with_list_input():    
    gao_it = GaoIt()
    # Test adding by name with list of strings
    gao_it.add_by_name(model, ['kf', 'kr'])    
    assert gao_it.mask(model.parameters) == [True, True, False]

def test_gaoit_add_by_name_with_tuple_input():
    gao_it = GaoIt()
    # Test adding by name with tuple of strings
    gao_it.add_by_name(model, ('kr', 'A_0'))    
    assert gao_it.mask(model.parameters) == [False, True, True]    

if __name__ == '__main__':
    test_gaoit_init()
    test_gaoit_addbycall()
    test_gaoit_isub()
    test_gaoit_func_add_all_kinetic_params()
    test_gaoit_getitem()
    test_gaoit_iadd()
    test_gaoit_contains()
    test_gaoit_setitem()
    test_gaoit_names()
    test_gaoit_keys()
    test_gaoit_mask()
    test_gaoit_locs()
    test_gaoit_widths()
    test_gaoit_sampled_parameters()
    test_gaoit_add_all_nonkinetic_params()
    test_gaoit_add_by_name_with_string_input()
    test_gaoit_add_by_name_with_list_input()
    test_gaoit_add_by_name_with_tuple_input()


